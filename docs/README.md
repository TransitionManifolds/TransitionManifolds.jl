Build documentation locally: `julia --project make.jl`.

View documentation through a webserver: Install `LiveServer` and run `julia -e 'using LiveServer; serve(dir="build")'`.

Generate links in `CHANGELOG.md`:
```
Changelog.generate(
    Changelog.CommonMark(),
    "CHANGELOG.md";
    repo = "TransitionManifolds/TransitionManifolds.jl",
)
```
