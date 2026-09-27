# utils/auto_push.ps1
# Background daemon: automatically checks for git changes every 60 seconds, commits, and pushes.

Write-Output "Auto-push daemon active. Checking for git changes every 60 seconds..."

while ($true) {
    Start-Sleep -Seconds 60
    
    $status = git status --porcelain
    if ($status) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Output "[$timestamp] Changes detected. Staging and committing..."
        
        git add -A
        $staged = git diff --cached --name-only
        if ($staged) {
            git commit -m "Auto-sync: $timestamp"
            git push origin main
            Write-Output "[$timestamp] Pushed to GitHub successfully."
        }
    }
}
