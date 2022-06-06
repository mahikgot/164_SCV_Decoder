Steven Esguerra
2019-05959
CoE 164 - FWX

Mark Guiang
2018 - 07118
CoE 164 - FWX

Capstone Project

Steps to run (by OS)

---Windows---
0. Install Python version 3.9.2 or later
1. Run PowerShell 7 in the same directory of the code
2. Input the following commands

PS> $OutputEncoding = [Console]::OutputEncoding = (New-Object System.Text.UTF8Encoding $false)
PS> $env:PYTHONIOENCODING = "utf-8"
PS> Get-Content [input_filename.txt] | python3 coe164_se03.py | Out-File [output_filename.txt]

---Linux Distro/Mac OS/Unix-Based OS---
0. Install Python version 3.9.2 or later
1. Run Terminal
2. Input the following command

$ python3 coe164_se01.py < [input_filename.txt] > [output_filename.txt]

Notes:
*It's more convenient if the input file is in the same location as the code. If its not possible, enter the exact location of the input file
*Output file will always generate in the same folder of the code (unless specified)
*If python3 is not available in your machine try entering python or py
