Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\fhard\GitHub\Project_Eva_AI\Fate_Desktop_MCP"
WshShell.Run "py -3.14 server.py", 0, False
