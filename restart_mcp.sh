#!/usr/bin/env bash
# Detached restarter for the macOS/Linux instances.
#
# MUST NOT be run as a child of the server it is about to kill -- unloading the
# LaunchAgent kills the whole job, including any shell the server spawned, so a
# naive "unload; load" from run_command_bg dies after the unload and never
# reaches the load, leaving the host with no MCP at all and no way back in.
# Launch it detached instead:
#     nohup ~/CloudSystems/PC_Use_MCP/restart_mcp.sh >/dev/null 2>&1 &
set -u
PLIST="$HOME/Library/LaunchAgents/com.fate.pcusemcp.plist"
sleep 3
if [ -f "$PLIST" ]; then
    launchctl unload "$PLIST" 2>/dev/null
    sleep 2
    launchctl load "$PLIST"
else
    systemctl --user restart pc-use-mcp.service
fi
