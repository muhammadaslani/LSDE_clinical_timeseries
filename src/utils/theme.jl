# Define the Atom One Dark colors
atom_one_dark = Dict(
    :background => colorant"#282C34",
    :foreground => colorant"#ABB2BF",
    :red        => colorant"#E06C75",
    :green      => colorant"#98C379",
    :yellow     => colorant"#E5C07B",
    :blue       => colorant"#61AFEF",
    :purple     => colorant"#C678DD",
    :cyan       => colorant"#56B6C2",
    :gray       => colorant"#5C6370"
)

# Create a color vector for easy access in plots
atom_one_dark_palette = [
    atom_one_dark[:purple],
    atom_one_dark[:cyan],
    atom_one_dark[:blue],
    atom_one_dark[:green],
    atom_one_dark[:yellow],
    atom_one_dark[:red]
]

# Function to get n colors from the palette (cycling if needed)
function get_atom_one_dark_colors(n::Int)
    return [atom_one_dark_palette[mod1(i, length(atom_one_dark_palette))] for i in 1:n]
end

# Set up a theme for Makie

atom_one_dark_theme = Theme(
    backgroundcolor = atom_one_dark[:background],
    textcolor = atom_one_dark[:foreground],
    palette = (color = atom_one_dark_palette,),
    Axis = (
        xgridvisible = false,
        ygridvisible = false,
        leftspinevisible = true,
        rightspinevisible = false,
        bottomspinevisible = true,
        topspinevisible = false,
        xminorticksvisible = false,
        yminorticksvisible = false,
        backgroundcolor = :transparent,
    )
)

# To use this theme globally:
set_theme!(atom_one_dark_theme)