powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.dropbox.com/s/o4t159po38tgtqz/voc2007train.zip?dl=1','%TEMP%\voc2007train.zip')"
powershell -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\voc2007train.zip', '.'); }"
IF EXIST "%TEMP%\voc2007train.zip" (
	DEL "%TEMP%\voc2007train.zip"
)
