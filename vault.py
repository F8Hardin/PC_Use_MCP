"""Obsidian vault tools for PC Use MCP.

Gives an agent structured read/write access to an Obsidian vault living on this
machine. Registered only when VAULT_ROOT is set, so the same codebase running
somewhere without a vault simply does not expose these tools.

Design notes
------------
* **No Obsidian dependency.** Everything is plain file I/O. The Local REST API
  would give full-text search and heading-level PATCH, but at this vault's scale
  (~130 notes, <5 MB) a Python scan is already sub-second, and dropping the API
  means the tools keep working while Obsidian is closed. Fewer moving parts.

* **Reads are wide, writes are narrow, and the narrowing is negotiable.** The
  agent may read the whole vault but writes only inside its own namespace
  (VAULT_AGENT_NS, plus Inbox/) unless the operator grants a specific path in
  the grants file. Enforced here rather than in a prompt, because a prompt rule
  is a suggestion. Grants are re-read per write, so permission can be given
  mid-conversation without a restart.

* **Writes are structured, not freeform.** vault_write_note builds the note --
  frontmatter, timestamps, ownership -- from arguments. A small local model
  hand-writing YAML produces malformed notes that poison search and the graph.
  It also refuses a note with no links, since an unreachable note is one the
  agent will never find again.

* **Search returns snippets, never bodies.** Recall is cheap; a tool that can
  dump 130 notes into a 75k context is not.
"""

import os
import re
import json
from datetime import datetime

import yaml

VAULT_ROOT = os.path.realpath(os.getenv("VAULT_ROOT", "").strip()) if os.getenv("VAULT_ROOT", "").strip() else ""
AGENT_NS = os.getenv("VAULT_AGENT_NS", "Eva").strip().strip("/\\")
AGENT_OWNER = os.getenv("VAULT_AGENT_OWNER", AGENT_NS.lower()).strip()
# Folders the agent may write to, beyond its own namespace.
SHARED_WRITE = ("Inbox",)

# Paths outside the namespace that the operator has granted. Read fresh on every
# write attempt rather than cached at startup, so a grant takes effect
# immediately instead of needing a server restart -- the point is to be able to
# say "yes, go ahead" mid-conversation. This is the minimal form of the
# "allow once / always allow" permission broker described in NOTES.txt.
GRANTS_FILE = os.getenv("VAULT_GRANTS_FILE", "").strip() or (
    os.path.join(VAULT_ROOT, ".vault_grants") if VAULT_ROOT else ""
)

VALID_FILECLASS = (
    "PersonFile", "PetFile", "OrgFile", "ProjectFile", "ToolFile", "ServerFile",
    "RepoFile", "SkillFile", "SessionFile", "AnalysisFile", "MocFile", "ServiceFile",
)

# Master nodes. Every note has to reach the graph, so one whose supplied links
# all dangle gets wired to the index for its folder, deepest match first, with
# the agent's home note as the floor. This is enforcement, not advice: the model
# that forgets to link is the same model that would have forgotten to read a
# rule telling it to link.
HUB_NOTE = os.getenv("VAULT_AGENT_HUB", "").strip() or AGENT_NS.lower()
FOLDER_HUBS = {
    f"{AGENT_NS}/Memory": f"{AGENT_NS}'s Memory",
    f"{AGENT_NS}/Journal": f"{AGENT_NS}'s Memory",
}

MAX_SNIPPET = 240
MAX_BODY_RETURN = 20000


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _rel(path: str) -> str:
    """Normalize a vault-relative path: forward slashes, no surrounding slashes
    or whitespace. Stripping whitespace matters -- paths arrive from grants-file
    lines and model output, both of which carry stray newlines and spaces that
    would otherwise silently fail every comparison."""
    return path.replace("\\", "/").strip().strip("/").strip()


def _resolve(path: str):
    """(abs_path, None) if inside the vault, else (None, error).

    realpath first so symlinks and .. cannot escape.
    """
    if not VAULT_ROOT:
        return None, "Vault tools are not configured (VAULT_ROOT is unset)."
    rel = _rel(path)
    if not rel:
        return None, "Empty path."
    if not rel.lower().endswith(".md"):
        rel += ".md"
    target = os.path.realpath(os.path.join(VAULT_ROOT, rel))
    if target != VAULT_ROOT and not target.startswith(VAULT_ROOT + os.sep):
        return None, f"Path {path!r} escapes the vault root."
    return target, None


def _load_grants():
    """Vault-relative path prefixes the operator has granted write access to.

    One per line, '#' comments and blanks ignored. A line may name a folder
    ('Sessions') or a single note ('Projects/Eva AI.md'). Missing file = no
    grants, which is the correct default.
    """
    if not GRANTS_FILE or not os.path.isfile(GRANTS_FILE):
        return []
    try:
        with open(GRANTS_FILE, encoding="utf-8") as f:
            return [
                _rel(line.split("#", 1)[0]).lower()
                for line in f
                if line.split("#", 1)[0].strip()
            ]
    except OSError:
        return []


def _writable(path: str):
    """(canonical_rel, error, grant) -- grant is the line that authorized this.

    Allowed in three cases: inside the agent's own namespace, inside a folder in
    SHARED_WRITE, or covered by an operator grant. The first two are standing
    permissions and return grant=None. The third returns the matching grant line,
    which the caller must pass to _consume_grant() once the write succeeds --
    grants are single-use.

    Namespace matching is case-insensitive so a model writing 'eva/' is not
    denied a path that is genuinely inside its namespace -- but the returned path
    is rewritten to the canonical casing. Permitting the lowercase spelling as-is
    would be harmless on Windows and quietly wrong on a case-sensitive
    filesystem, where it would fork a second 'eva/' folder alongside 'Eva/'.
    """
    rel = _rel(path)
    parts = rel.split("/")
    top = parts[0].lower()

    if top == AGENT_NS.lower():
        return "/".join([AGENT_NS] + parts[1:]), None, None
    shared = next((s for s in SHARED_WRITE if s.lower() == top), None)
    if shared:
        return "/".join([shared] + parts[1:]), None, None

    low = rel.lower()
    for g in _load_grants():
        if low == g or low.startswith(g + "/") or low == g + ".md":
            return rel, None, g

    return None, (
        f"Write denied: {rel!r} is outside your namespace and has not been granted. "
        f"You may always write under '{AGENT_NS}/' and append to {list(SHARED_WRITE)}. "
        f"To edit this note, ask Fate to grant it -- he adds a line to "
        f"{os.path.basename(GRANTS_FILE) or '.vault_grants'} in the vault root. "
        f"Say which path you want and why; do not try to route around this. "
        f"A grant covers one write only, so ask again for the next one."
    ), None


def _consume_grant(grant: str) -> bool:
    """Remove a used grant from the grants file. Returns True if it was removed.

    Grants are single-use: one grant authorizes one write, then it is spent. The
    operator re-grants for the next one. Called only AFTER the write succeeds, so
    a failed write does not burn the permission.

    Removes the first matching line and leaves comments and other grants intact.
    """
    if not grant or not GRANTS_FILE or not os.path.isfile(GRANTS_FILE):
        return False
    try:
        with open(GRANTS_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            body = line.split("#", 1)[0]
            if body.strip() and _rel(body).lower() == grant:
                del lines[i]
                with open(GRANTS_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                return True
    except OSError:
        return False
    return False


def _iter_notes():
    """Yield (abs_path, vault_relative_path) for every note, skipping .obsidian."""
    for dirpath, dirnames, filenames in os.walk(VAULT_ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                ap = os.path.join(dirpath, fn)
                yield ap, _rel(os.path.relpath(ap, VAULT_ROOT))


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
_LINK = re.compile(r"\[\[([^\]|#]+)")
_FENCE = re.compile(r"^```.*?^```", re.S | re.M)
_CODESPAN = re.compile(r"`[^`\n]*`")


def _split(text: str):
    """(frontmatter_dict, body). Malformed YAML degrades to {} rather than raising."""
    m = _FM.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    return fm, text[m.end():]


def _compose(fm: dict, body: str) -> str:
    return "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip() + "\n---\n\n" + body.lstrip("\n")


def _links_in(text: str):
    """Wikilink targets, de-duplicated, order preserved.

    Code spans and fenced blocks are stripped first, matching Obsidian: a
    wikilink inside backticks is displayed as text and creates no graph edge.
    Without this, a note that *documents* a malformed link acquires it.
    """
    text = _FENCE.sub(" ", text)
    text = _CODESPAN.sub(" ", text)
    seen, out = set(), []
    for raw in _LINK.findall(text):
        t = raw.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _title_of(rel: str) -> str:
    return os.path.basename(rel)[:-3]


# Keys a model plausibly wraps a note title in when it sends an object instead
# of a string. Ordered by how unambiguous each one is.
_TITLE_KEYS = ("title", "name", "note", "target", "link", "tag", "path", "value")


def _as_title(v) -> str:
    """Coerce one links/tags element to a plain note title.

    Small local models routinely send `[{"title": "Moltbook"}]` where the schema
    wants `["Moltbook"]`. MCP declares both parameters as a bare `list`, so
    nothing rejects it, and the old `str(v)` baked `{'title': 'Moltbook'}`
    straight into a wikilink. The result is a note that *looks* linked, passes
    the "at least one link" check, and is in fact an orphan pointing at a note
    that can never exist -- exactly the failure this module was written to
    prevent, arriving through the door left open for it.

    Also accepts a path (`Projects/Eva AI.md`) and reduces it to the title, since
    that is the other shape models reach for. Returns "" for anything it cannot
    read, so the caller drops it rather than writing a garbage link.
    """
    if isinstance(v, dict):
        for k in _TITLE_KEYS:
            if isinstance(v.get(k), str) and v[k].strip():
                v = v[k]
                break
        else:
            # Unrecognized shape: salvage it only if there is exactly one string
            # in it, so we never silently pick the wrong field.
            vals = [x for x in v.values() if isinstance(x, str) and x.strip()]
            if len(vals) != 1:
                return ""
            v = vals[0]
    elif isinstance(v, (list, tuple)):
        return _as_title(v[0]) if len(v) == 1 else ""

    if not isinstance(v, str):
        return ""
    t = v.strip().strip("[]").strip()
    if t.lower().endswith(".md"):
        t = _title_of(_rel(t))
    return t.strip()


def _read(ap: str) -> str:
    with open(ap, encoding="utf-8") as f:
        return f.read()


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_vault_tools(mcp):
    """Attach the vault tools to a FastMCP instance. No-op without VAULT_ROOT."""
    if not VAULT_ROOT:
        print("[vault] VAULT_ROOT unset - vault tools not registered.")
        return False
    if not os.path.isdir(VAULT_ROOT):
        print(f"[vault] VAULT_ROOT {VAULT_ROOT!r} is not a directory - vault tools not registered.")
        return False

    @mcp.tool()
    def vault_search(query: str, limit: int = 10) -> str:
        """
        Search the whole vault for text and return matching notes as paths with
        short snippets. This is the cheapest way to find knowledge -- start here.

        Matches note titles and body text, case-insensitively. Title matches rank
        first, then notes with more hits. Returns snippets only, never full notes;
        call vault_read on a path once you know which note you want.

        query: words to look for, e.g. 'moltbook posting' or 'beaker'
        limit: max notes to return (default 10)
        """
        if not query.strip():
            return _err("Empty query.")
        q = query.lower()
        terms = [t for t in re.split(r"\s+", q) if t]
        hits = []
        for ap, rel in _iter_notes():
            try:
                text = _read(ap)
            except OSError:
                continue
            low = text.lower()
            title = _title_of(rel)
            tl = title.lower()
            # Score: whole-phrase title hit > term-in-title > body frequency.
            score = 0
            if q in tl:
                score += 100
            score += 20 * sum(1 for t in terms if t in tl)
            body_hits = sum(low.count(t) for t in terms)
            score += min(body_hits, 20)
            if score == 0:
                continue
            pos = next((low.find(t) for t in terms if t in low), -1)
            if pos < 0:
                snippet = text[:MAX_SNIPPET]
            else:
                start = max(0, pos - 80)
                snippet = ("..." if start else "") + text[start:start + MAX_SNIPPET].strip()
            hits.append({
                "path": rel,
                "title": title,
                "score": score,
                "snippet": " ".join(snippet.split()),
            })
        hits.sort(key=lambda h: -h["score"])
        hits = hits[:max(1, limit)]
        for h in hits:
            h.pop("score")
        return json.dumps({"query": query, "count": len(hits), "results": hits}, ensure_ascii=False)

    @mcp.tool()
    def vault_read(path: str) -> str:
        """
        Read one note in full: its frontmatter, its body, and the notes it links to.

        path: vault-relative, e.g. 'Eva/eva.md' or 'Projects/Eva AI'
              (the .md extension is optional)
        """
        ap, err = _resolve(path)
        if err:
            return _err(err)
        if not os.path.isfile(ap):
            return _err(f"No note at {_rel(path)!r}. Use vault_search to find the right path.")
        try:
            text = _read(ap)
        except OSError as e:
            return _err(f"Could not read {_rel(path)!r}: {e}")
        fm, body = _split(text)
        truncated = len(body) > MAX_BODY_RETURN
        return json.dumps({
            "path": _rel(path),
            "title": _title_of(_rel(ap.replace(VAULT_ROOT, "").lstrip(os.sep)) or path),
            "frontmatter": fm,
            "links": _links_in(body),
            "body": body[:MAX_BODY_RETURN] + ("\n\n[...truncated]" if truncated else ""),
            "truncated": truncated,
        }, ensure_ascii=False, default=str)

    @mcp.tool()
    def vault_links(path: str) -> str:
        """
        Show what a note links to (outgoing) and what links back to it (backlinks).

        Following links is often faster than searching again, and surfaces related
        context you did not know to ask for. Use it after vault_read to widen out
        from a note that turned out to be relevant.

        path: vault-relative path to the note
        """
        ap, err = _resolve(path)
        if err:
            return _err(err)
        if not os.path.isfile(ap):
            return _err(f"No note at {_rel(path)!r}.")
        title = _title_of(_rel(os.path.relpath(ap, VAULT_ROOT)))
        try:
            outgoing = _links_in(_split(_read(ap))[1])
        except OSError as e:
            return _err(f"Could not read {_rel(path)!r}: {e}")

        # Resolve outgoing targets to real paths so unresolved ones are visible.
        by_title = {}
        for oap, orel in _iter_notes():
            by_title.setdefault(_title_of(orel), orel)

        resolved, unresolved = [], []
        for t in outgoing:
            (resolved if t in by_title else unresolved).append(by_title.get(t, t))

        backlinks = []
        for oap, orel in _iter_notes():
            if orel == _rel(os.path.relpath(ap, VAULT_ROOT)):
                continue
            try:
                if title in _links_in(_read(oap)):
                    backlinks.append(orel)
            except OSError:
                continue

        return json.dumps({
            "path": _rel(path),
            "title": title,
            "links_to": resolved,
            "links_to_missing": unresolved,
            "linked_from": backlinks,
        }, ensure_ascii=False)

    @mcp.tool()
    def vault_query(fileClass: str = "", type: str = "", owner: str = "",
                    tag: str = "", status: str = "", folder: str = "", limit: int = 50) -> str:
        """
        Find notes by their frontmatter rather than their text. Use when you want
        a category instead of a topic -- every person, every active project,
        everything you wrote.

        All filters are optional and combine with AND. Leave one empty to ignore it.

        fileClass: PersonFile, PetFile, OrgFile, ProjectFile, ToolFile, ServerFile,
                   RepoFile, SkillFile, SessionFile, AnalysisFile, MocFile, ServiceFile
        type:      project, reference, session, inbox, moc, skill, ai_agent_skill, analysis
        owner:     e.g. 'eva' or 'claude'
        tag:       a single tag to require
        status:    active, paused, done
        folder:    restrict to a top-level folder, e.g. 'Eva' or 'Projects'
        limit:     max notes to return (default 50)
        """
        want = {"fileClass": fileClass, "type": type, "owner": owner, "status": status}
        want = {k: v.strip().lower() for k, v in want.items() if v and v.strip()}
        tag = tag.strip().lower()
        folder = _rel(folder)

        out = []
        for ap, rel in _iter_notes():
            if folder and not (rel == folder or rel.startswith(folder + "/")):
                continue
            try:
                fm, _ = _split(_read(ap))
            except OSError:
                continue
            if any(str(fm.get(k, "")).strip().lower() != v for k, v in want.items()):
                continue
            if tag:
                tags = fm.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]
                if tag not in [str(t).strip().lower() for t in tags]:
                    continue
            out.append({
                "path": rel,
                "title": _title_of(rel),
                "fileClass": fm.get("fileClass"),
                "type": fm.get("type"),
                "status": fm.get("status"),
                "updated": str(fm.get("updated")) if fm.get("updated") else None,
            })
            if len(out) >= max(1, limit):
                break
        return json.dumps({"count": len(out), "results": out}, ensure_ascii=False)

    @mcp.tool()
    def vault_write_note(title: str, fileClass: str, body: str, links: list = [],
                         tags: list = [], folder: str = "", type: str = "",
                         status: str = "", overwrite: bool = False) -> str:
        """
        Create or replace a note. Frontmatter, dates, and ownership are filled in
        for you -- supply the content and the connections.

        title:     specific enough to recognize later. Becomes the filename.
        fileClass: what KIND of thing the note is about -- PersonFile, PetFile,
                   OrgFile, ProjectFile, ToolFile, ServerFile, RepoFile, SkillFile,
                   SessionFile, AnalysisFile, MocFile, ServiceFile
        body:      the note in markdown. One topic per note -- if you are writing
                   about two things, write two notes and link them.
        links:     REQUIRED, at least one. Titles of related notes, e.g.
                   ['Eva AI', 'Fate (F8Hardin)']. A note nothing links to is a note
                   you will never find again. Link to hubs, people, and projects.
        tags:      lowercase, reuse existing ones where they fit.
        folder:    defaults to your namespace. Must be inside it.
        type:      project, reference, session, moc, analysis, ai_agent_skill...
        status:    active, paused, done -- omit if not applicable.
        overwrite: refuse to clobber an existing note unless True. If you are
                   adding to a note that exists, prefer vault_append.
        """
        title = (title or "").strip()
        if not title:
            return _err("A title is required.")
        bad = set('<>:"/\\|?*')
        if bad & set(title):
            return _err(f"Title contains characters not allowed in a filename: {''.join(sorted(bad & set(title)))}")
        if fileClass not in VALID_FILECLASS:
            return _err(f"fileClass must be one of {list(VALID_FILECLASS)}, got {fileClass!r}.")
        if isinstance(links, str):
            links = [links]
        links = [t for t in (_as_title(l) for l in (links or [])) if t]
        if not links:
            return _err(
                "At least one link is required. Name a related note in `links` -- "
                "a hub like 'Eva AI', the person involved, or the project it touches. "
                "Pass plain title strings: links=['Moltbook', 'Eva AI']. Not objects, "
                "not {'title': 'Moltbook'}. "
                "Use vault_search first if you are unsure what exists; a link to a note "
                "that does not exist yet is fine and marks it as worth writing."
            )
        if isinstance(tags, str):
            tags = [tags]
        tags = [t.lower().lstrip("#") for t in (_as_title(x) for x in (tags or [])) if t]

        rel = _rel(folder) if folder else AGENT_NS
        rel, werr, grant = _writable(f"{rel}/{title}.md")
        if werr:
            return _err(werr)
        ap, err = _resolve(rel)
        if err:
            return _err(err)
        if os.path.exists(ap) and not overwrite:
            return _err(
                f"A note already exists at {_rel(rel)!r}. Use vault_append to add to it, "
                f"or pass overwrite=True to replace it deliberately."
            )

        today = datetime.now().strftime("%Y-%m-%d")
        existing_created = None
        if os.path.exists(ap):
            try:
                existing_created = _split(_read(ap))[0].get("created")
            except OSError:
                pass

        fm = {"type": (type or "").strip() or "reference", "fileClass": fileClass, "owner": AGENT_OWNER}
        if status.strip():
            fm["status"] = status.strip()
        fm["created"] = str(existing_created) if existing_created else today
        fm["updated"] = today
        fm["tags"] = tags or [AGENT_OWNER]

        # Reaching the graph means linking to a note that actually exists. A link
        # to a not-yet-written note is legitimate and stays, but it cannot be the
        # note's only tie -- that is an orphan wearing a link. If every supplied
        # link dangles, wire it to its folder's master node as well.
        existing = {_title_of(r) for _, r in _iter_notes()}
        unresolved = [l for l in links if l not in existing]
        folder_rel = os.path.dirname(_rel(rel))
        hub = next(
            (h for f, h in FOLDER_HUBS.items() if folder_rel.lower() == f.lower()),
            HUB_NOTE,
        )
        hubbed = None
        if len(unresolved) == len(links) and hub not in links:
            links.append(hub)
            hubbed = hub

        # Guarantee the links are actually present in the note text, so the graph
        # sees them even if the model forgot to write them into the body.
        body = (body or "").strip()
        present = set(_links_in(body))
        missing = [l for l in links if l not in present]
        if missing:
            body += "\n\n---\nRelated: " + " · ".join(f"[[{l}]]" for l in missing) + "\n"

        os.makedirs(os.path.dirname(ap), exist_ok=True)
        try:
            with open(ap, "w", encoding="utf-8") as f:
                f.write(_compose(fm, f"# {title}\n\n{body}\n" if not body.startswith("#") else body + "\n"))
        except OSError as e:
            return _err(f"Could not write {_rel(rel)!r}: {e}")
        out = {"ok": True, "path": _rel(rel), "frontmatter": fm, "links": links}
        if unresolved:
            out["links_unresolved"] = unresolved
            out["note"] = (
                f"These links point at notes that do not exist yet: {unresolved}. "
                f"That is fine if you meant to flag them as worth writing. If you "
                f"meant an existing note, the title is probably spelled differently "
                f"-- vault_search for it and rewrite with the real title."
            )
        if hubbed:
            out["hub_added"] = hubbed
            out["note"] = (
                f"Every link you gave was to a note that does not exist, which would "
                f"have left this note unreachable, so it was also linked to [[{hubbed}]]. "
                f"Link to a real note next time -- vault_search first to get the title right."
            )
        if grant and _consume_grant(grant):
            out["grant_consumed"] = (
                f"The grant for {grant!r} was single-use and is now spent. "
                f"Ask Fate again before writing outside your namespace next time."
            )
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def vault_append(path: str, text: str, heading: str = "") -> str:
        """
        Add text to an existing note without rewriting it. Prefer this over
        vault_write_note when the note already exists -- it preserves what is
        there and bumps `updated` for you.

        path:    vault-relative path to the note
        text:    markdown to add
        heading: optional. Append under this heading (e.g. 'Evidence log').
                 Text lands at the end of that section, before the next heading.
                 If the heading is absent the note gains it at the end.
        """
        if not (text or "").strip():
            return _err("Nothing to append.")
        path, werr, grant = _writable(path)
        if werr:
            return _err(werr)
        ap, err = _resolve(path)
        if err:
            return _err(err)
        if not os.path.isfile(ap):
            return _err(f"No note at {_rel(path)!r}. Use vault_write_note to create it.")
        try:
            fm, body = _split(_read(ap))
        except OSError as e:
            return _err(f"Could not read {_rel(path)!r}: {e}")

        text = text.strip()
        if not heading.strip():
            body = body.rstrip() + "\n\n" + text + "\n"
            where = "end of note"
        else:
            h = heading.strip().lstrip("#").strip()
            lines = body.split("\n")
            idx = next((i for i, l in enumerate(lines)
                        if l.startswith("#") and l.lstrip("#").strip().lower() == h.lower()), None)
            if idx is None:
                body = body.rstrip() + f"\n\n## {h}\n\n{text}\n"
                where = f"new section '{h}'"
            else:
                level = len(lines[idx]) - len(lines[idx].lstrip("#"))
                end = len(lines)
                for j in range(idx + 1, len(lines)):
                    if lines[j].startswith("#"):
                        if (len(lines[j]) - len(lines[j].lstrip("#"))) <= level:
                            end = j
                            break
                while end > idx + 1 and not lines[end - 1].strip():
                    end -= 1
                lines[end:end] = ["", text]
                body = "\n".join(lines)
                where = f"section '{h}'"

        fm["updated"] = datetime.now().strftime("%Y-%m-%d")
        try:
            with open(ap, "w", encoding="utf-8") as f:
                f.write(_compose(fm, body))
        except OSError as e:
            return _err(f"Could not write {_rel(path)!r}: {e}")
        out = {"ok": True, "path": _rel(path), "appended_to": where}
        if grant and _consume_grant(grant):
            out["grant_consumed"] = (
                f"The grant for {grant!r} was single-use and is now spent. "
                f"Ask Fate again before writing outside your namespace next time."
            )
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def vault_list(folder: str = "") -> str:
        """
        List notes and subfolders. Use to get oriented; use vault_search to find
        something specific.

        folder: vault-relative folder, e.g. 'Eva' or 'Projects'. Empty = vault root.
        """
        rel = _rel(folder)
        target = os.path.realpath(os.path.join(VAULT_ROOT, rel)) if rel else VAULT_ROOT
        if target != VAULT_ROOT and not target.startswith(VAULT_ROOT + os.sep):
            return _err(f"Path {folder!r} escapes the vault root.")
        if not os.path.isdir(target):
            return _err(f"No folder at {rel!r}.")
        folders, notes = [], []
        for name in sorted(os.listdir(target)):
            if name.startswith("."):
                continue
            full = os.path.join(target, name)
            if os.path.isdir(full):
                folders.append(name)
            elif name.lower().endswith(".md"):
                notes.append({"path": f"{rel}/{name}" if rel else name, "title": name[:-3]})
        return json.dumps({"folder": rel or "/", "folders": folders, "notes": notes,
                           "writable": _writable(rel or AGENT_NS)[1] is None}, ensure_ascii=False)
        # note: index [1] is the error slot; [2] is the grant, irrelevant for a listing

    print(f"[vault] tools registered - root={VAULT_ROOT} namespace={AGENT_NS}/ owner={AGENT_OWNER}")
    return True
