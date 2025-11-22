# Script to clean stale Jupyter kernel files
# Run this if you experience "Kernel does not exist" errors

$runtimeDir = "$env:APPDATA\jupyter\runtime"
$kernelFiles = Get-ChildItem "$runtimeDir\kernel-*.json" -ErrorAction SilentlyContinue

if ($kernelFiles) {
    Write-Host "Found $($kernelFiles.Count) kernel files"
    
    # Remove kernel files older than 1 hour (stale kernels)
    $oldKernels = $kernelFiles | Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) }
    if ($oldKernels) {
        $oldKernels | Remove-Item -Force
        Write-Host "Cleaned $($oldKernels.Count) stale kernel files"
    } else {
        Write-Host "No stale kernels found (all are recent)"
    }
    
    # Show remaining kernels
    $remaining = Get-ChildItem "$runtimeDir\kernel-*.json" -ErrorAction SilentlyContinue
    Write-Host "Remaining kernel files: $($remaining.Count)"
} else {
    Write-Host "No kernel files found"
}

Write-Host "`nTo completely reset, you can also:"
Write-Host "1. Close all Jupyter notebooks"
Write-Host "2. Stop Jupyter Lab server (Ctrl+C)"
Write-Host "3. Delete all files in: $runtimeDir"
Write-Host "4. Restart Jupyter Lab"

