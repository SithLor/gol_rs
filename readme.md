## Setup Instructions (Windows)

### 1. Install Rust
Download and run the installer from:
https://www.rust-lang.org/tools/install

After installation, open a new terminal and run:
```
rustup install nightly
```

### 2. Install Chocolatey (Windows Package Manager)
Open PowerShell as Administrator and run:
```
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('http://internal/odata/repo/ChocolateyInstall.ps1'))
```

### 3. Install ffmpeg (using Chocolatey)
In the same PowerShell window, run:
```
choco install ffmpeg.full
```


### to run
./start.ps1 -rust or -java or -both





