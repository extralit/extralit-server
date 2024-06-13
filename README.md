
<h1 align="center">
  <a href=""><img src="https://github.com/dvsrepo/imgs/raw/main/rg.svg" alt="Argilla" width="150"></a>
  <br>
  Extralit
  <br>
</h1>

<h2 align="center">Open-source feedback layer for LLM-assisted data extractions</h2>

<h3>
<p align="center">
<a href="https://docs.argilla.io">📄 Documentation</a> | </span>
<a href="#-quickstart">🚀 Quickstart</a> <span> | </span>
<a href="#-project-architecture">🛠️ Architecture</a> <span> | </span>
</p>
</h3>

## What is Extralit?

Extralit is a UI interface and platform for LLM-based document data extraction that integrates human and model feedback loops for continuous LLM refinement and data extraction oversight.

With a Python SDK and flexible UI, you can create human and model-in-the-loop workflows for:

* Data extraction validation
* Supervised fine-tuning
* Preference tuning (RLHF, DPO, RLAIF, and more)
* Small, specialized NLP models
* Scalable evaluation.

## 🚀 Development Quickstart

### Install the Pre-requisites
These steps are required to run and develop Argilla locally.

1. Install [Docker Desktop](https://docs.docker.com/get-docker/)
2. Install [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
2. Install [ctlptl](https://github.com/tilt-dev/ctlptl/tree/main#how-do-i-install-it)
3. Install [Tilt](https://docs.tilt.dev/)

### Set up local infrastructure for Kind

1. Create a `kind` cluster

```bash
ctlptl create registry ctlptl-registry --port=5005
ctlptl create cluster kind --registry=ctlptl-registry
```


2. Apply config to mount local directory

```bash
ctlptl apply -f k8s/kind/kind-config.yaml
kubectl taint node kind-control-plane node-role.kubernetes.io/control-plane:NoSchedule-

```

### Start local development

1. Run Tilt 

Select the K8s cluster
```bash
kubectl config set-cluster <cluster_name>
```

Setting the `ENV` variable to `dev` enables hot-reloading of Docker containers for 🚀 rapid deployment:
```bash
kubectl create ns <namespace>
ENV=dev tilt up --namespace=<namespace>
```

### Start staging/prod K8s deployment

```bash
ENV=dev DOCKER_REPO=<remote docker repository> tilt up --namespace <namespace> --context <K8s cluster context>
```

## 🛠️ Developer guide

### Editing database schema:
Editting the database schema files at `src/argilla/server/models/*.py` require running these commands to apply revisions to the database.

1. Create revision
```bash
cd src/argilla
alembic revision -m <message>
```

If you happen to run into errors due to the revisions from upstream argilla-io/argilla repo, set the down-revision tag to their latest in the revision `"7552df94427a"` at `src/argilla/server/alembic/versions`

2. Apply the revision
```bash
# Be sure to set environment variables ARGILLA_ELASTICSEARCH and ARGILLA_DATABASE_URL
python -m argilla server database migrate
```

3. Update frontend site to the API backend

```bash
bash scripts/build_frontend.sh
python setup.py bdist_wheel
```

## 🛠️ Project Architecture

Argilla is built on 5 core components:

- **Python SDK**: A Python SDK which is installable with `pip install argilla`. To interact with the Argilla Server and the Argilla UI. It provides an API to manage the data, configuration and annotation workflows.
- **FastAPI Server**: The core of Argilla is a *Python FastAPI* server that manages the data, by pre-processing it and storing it in the vector database. Also, it stores application information in the relational database. It provides a REST API to interact with the data from the Python SDK and the Argilla UI. It also provides a web interface to visualize the data.
- **Relational Database**: A relational database to store the metadata of the records and the annotations. *SQLite* is used as the default built-in option and is deployed separately with the Argilla Server but a separate *PostgreSQL* can be used too.
- **Vector Database**: A vector database to store the records data and perform scalable vector similarity searches and basic document searches. We currently support *ElasticSearch* and *AWS OpenSearch* and they can be deployed as separate Docker images.
- **Vue.js UI**: A web application to visualize and annotate your data, users and teams. It is built with *Vue.js* and is directly deployed alongside the Argilla Server within our Argilla Docker image.



<p align="center">
<a  href="https://pypi.org/project/argilla-server/">
<img alt="CI" src="https://img.shields.io/pypi/v/argilla.svg?style=flat-round&logo=pypi&logoColor=white">
</a>
<img alt="Codecov" src="https://codecov.io/gh/argilla-io/argilla-server/branch/main/graph/badge.svg?token=VDVR29VOMG"/>
<a href="https://pepy.tech/project/argilla-server">
<img alt="CI" src="https://static.pepy.tech/personalized-badge/argilla-server?period=month&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads/month">
</a>
<a href="https://huggingface.co/new-space?template=argilla/argilla-template-space">
<img src="https://huggingface.co/datasets/huggingface/badges/raw/main/deploy-to-spaces-sm.svg"/>
</a>
</p>

<p align="center">
<a href="https://twitter.com/argilla_io">
<img src="https://img.shields.io/badge/twitter-black?logo=x"/>
</a>
<a href="https://www.linkedin.com/company/argilla-io">
<img src="https://img.shields.io/badge/linkedin-blue?logo=linkedin"/>
</a>
<a href="https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g">
<img src="https://img.shields.io/badge/slack-purple?logo=slack"/>
</a>
</p>


## Clone repository

`argilla-server` is using `argilla` repository as submodule to build frontend statics so when cloning use the following command:

```sh
git clone --recurse-submodules git@github.com:argilla-io/argilla-server.git
```

If you already cloned the repository without using `--recurse-submodules` you can init and update the submodules with:

```sh
git submodule update --remote --recursive --init
```

> [!IMPORTANT]
> By default `argilla` submodule is using `develop` branch so the previous command will get the latest commit from that branch.

### Specify a tag for argilla submodule

When doing a release we should change `argilla` submodule to use an specific tag. In the following example we are setting tag `v1.22.0`:

```sh
cd argilla
git fetch --tags
git checkout v1.22.0
```

> [!NOTE]
> You should see some changes on the `argilla-server` root folder where the subproject commit is now changed to the one from the tag version. Feel free to commit these changes.

## Development environment

By default all commands executed with `pdm run` will get environment variables from `.env.dev` except command `pdm test` that will overwrite some of them using values coming from `.env.test` file.

These environment variables can be overrided if necessary so feel free to defined your own ones locally.

### Run cli

```sh
pdm cli
```

### Run database migrations

By default a SQLite located at `~/.argilla/argilla.db` will be used. You can create the database and run migrations with the following custom PDM command:

```sh
pdm migrate
```

### Run tests

A SQLite database located at `~/.argilla/argilla-test.db` will be automatically created to run tests. You can run the entire test suite using the following custom PDM command:

```sh
pdm test
```

## Run development server

### Build frontend static files

Before running Argilla development server we need to build the frontend static files. Node version 18 is required for this action:

```sh
brew install node@18
```

After that you can build the frontend static files:

```sh
./scripts/build_frontend.sh
```

After running the previous script you should have a folder at `src/argilla_server/static` with all the frontend static files successfully generated.

### Run uvicorn development server

```sh
pdm server
```
