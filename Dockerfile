FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 researcher
WORKDIR /work
COPY --from=build /src/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
USER researcher
VOLUME ["/work/results", "/work/data"]
ENTRYPOINT ["crypto-stat-arb"]
CMD ["--help"]
