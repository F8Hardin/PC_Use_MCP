' Starts server.py hidden. Works from wherever this repo is cloned.
' Copy it into shell:startup to run at logon.
Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "py server.py", 0, False
