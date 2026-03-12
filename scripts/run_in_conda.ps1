param(
    [Parameter(Mandatory = $true)]
    [string]$Module,

    [string]$EnvName = "math-tutor-agent",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$command = @("run", "-n", $EnvName, "python", "-m", $Module) + $Args
& conda @command
exit $LASTEXITCODE
