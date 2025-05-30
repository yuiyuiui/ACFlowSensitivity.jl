using ACFlowSensitivity
using Documenter

DocMeta.setdocmeta!(
    ACFlowSensitivity,
    :DocTestSetup,
    :(using ACFlowSensitivity);
    recursive = true,
)

makedocs(;
    modules = [ACFlowSensitivity],
    authors = "yuiyuiui",
    sitename = "ACFlowSensitivity.jl",
    format = Documenter.HTML(;
        canonical = "https://yuiyuiui.github.io/ACFlowSensitivity.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/yuiyuiui/ACFlowSensitivity.jl", devbranch = "main")
