jobs:
  benchmark:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/YOUR_GH_USER/petsc-python-miniapp:cpu
      # or: USERNAME/petsc-miniapp:cpu if using Docker Hub

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run PETSc benchmark (CPU)
        run: |
          python3 benchmark_petsc.py \
            --mat data/mat.dat \
            --rhs data/rhs.dat \
            --guess data/guess.dat \
            --ref data/sol.dat \
            --config data/options.json
