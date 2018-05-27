@echo off
powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/4x1m1lugp6lkctm/yolov3.weights?dl=1','darknet53.weights')"
pause