using Documenter
using ACFlowSensitivity

makedocs(
    sitename = "ACFlowSensitivity.jl",
    format = Documenter.HTML(),
    modules = [ACFlowSensitivity],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/yourusername/ACFlowSensitivity.jl.git"
) 