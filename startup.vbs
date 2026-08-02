Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\fhard\GitHub\PC_Use_MCP"
WshShell.Run "py -3.14 server.py", 0, False
