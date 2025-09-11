# ask to start rust or java version if they do ansew do both  cli ./start.ps1 -rust or -java or -both
param (
	[switch]$rust,
	[switch]$java,
	[switch]$both
)
$cwd = Get-Location

if ($rust -or $both) {
	Write-Host "Starting Rust version..."
	# Navigate to the Rust project directory
	# Run the Rust program
	cargo build --release
	.\target\release\gol_rs.exe
	# Return to the original directory
}
if ($java -or $both) {
	Write-Host "Starting Java version..."
	# Navigate to the Java project directory
	Set-Location -Path ".\java_src"
	# Run the Java program
	java ./Gol.java
	# Return to the original directory
	Set-Location -Path $cwd
}
