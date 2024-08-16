using Rhythm
using Documenter

DocMeta.setdocmeta!(Rhythm, :DocTestSetup, :(using Rhythm); recursive=true)

makedocs(;
    modules=[Rhythm],
    authors="Ahmed ElGazzar",
    sitename="Rhythm.jl",
    format=Documenter.HTML(;
        canonical="https://elgazzarr.github.io/Rhythm.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/elgazzarr/Rhythm.jl",
    devbranch="main",
)
