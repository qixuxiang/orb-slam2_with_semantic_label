powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/hl80jttpj0u6o87/voc2007valid.zip?dl=1','%TEMP%\voc2007valid.zip')"
powershell -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\voc2007valid.zip', '.'); }"
IF EXIST "%TEMP%\voc2007valid.zip" (
	DEL "%TEMP%\voc2007valid.zip"
)
