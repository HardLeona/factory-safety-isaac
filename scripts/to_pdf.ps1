# submission 폴더의 docx, pptx 를 PDF 로 바꾸고 쪽수를 알려준다 (Microsoft Word, PowerPoint 필요).
#   powershell -ExecutionPolicy Bypass -File scripts/to_pdf.ps1
$dir = Join-Path (Split-Path -Parent $PSScriptRoot) "submission"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
Get-ChildItem $dir -Filter *.docx | Where-Object { $_.Name -notlike "~$*" } | ForEach-Object {
    $doc = $word.Documents.Open($_.FullName, $false, $true)
    $pdf = [System.IO.Path]::ChangeExtension($_.FullName, ".pdf")
    $doc.SaveAs([ref]$pdf, [ref]17)
    $pages = $doc.ComputeStatistics(2)
    $doc.Close($false)
    Write-Output ("{0}: {1} pages" -f $_.Name, $pages)
}
$word.Quit()
$ppt = New-Object -ComObject PowerPoint.Application
Get-ChildItem $dir -Filter *.pptx | Where-Object { $_.Name -notlike "~$*" } | ForEach-Object {
    $p = $ppt.Presentations.Open($_.FullName, $true, $false, $false)
    $pdf = [System.IO.Path]::ChangeExtension($_.FullName, ".pdf")
    $p.SaveAs($pdf, 32)
    Write-Output ("{0}: {1} slides" -f $_.Name, $p.Slides.Count)
    $p.Close()
}
$ppt.Quit()
