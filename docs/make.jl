using Documenter
using ACFlowSensitivity

makedocs(; modules=[ACFlowSensitivity],
         sitename="ACFlowSensitivity.jl",
         authors="Kaiwen Jin and collaborators",
         pages=["Home" => "index.md",
                "Manual" => ["main/intro.md"]],
         format=Documenter.HTML())

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(; repo="github.com/yourusername/ACFlowSensitivity.jl.git")
