---
title: "Kubernetes for Data Scientists: Deploying a FastAPI App on GKE"
date: 2026-04-10
categories: [Infrastructure]
tags: [kubernetes, deployment, gcp, fastapi]
---

You've built a model, wrapped it in a FastAPI endpoint, and it works on your laptop. Now someone asks "can you deploy that?" and suddenly you're reading about pods, clusters, nodes, and ingress controllers. Kubernetes has a reputation for being complex, and honestly a lot of that reputation is deserved — but the core ideas are surprisingly simple once you strip away the jargon.

This post is a walkthrough aimed at people like me — data scientists who build APIs but haven't really touched infrastructure directly, maybe just dealing with Kubernetes through pre-defined CI/CD pipelines and a bit of debugging. We'll go from a FastAPI app to a running service on Google Kubernetes Engine (GKE), and I'll try to explain what each piece actually does along the way. All the code is in the [companion repo](https://github.com/acjones27/fastapi-k8s-example), including deploy and teardown scripts if you want to try it yourself.

---

## What Kubernetes actually is

Kubernetes (often abbreviated to K8s) is a system for running and managing containers across a cluster of machines. That's it. Everything else is details about how it does that.

The mental model that helped me: think of Kubernetes as a very persistent operations person. You tell it "I want three copies of my API running at all times, each with 512MB of memory" and it makes that happen. If a copy crashes, it starts a new one. If a machine goes down, it moves the workload elsewhere. If you push a new version, it rolls it out gradually so there's no downtime. You describe what you want, and Kubernetes figures out how to get there and stay there.

This is called **declarative configuration** — you write down the desired state in YAML files, and Kubernetes continuously works to make reality match that description. It's different from the imperative approach of "SSH into a machine and run these commands", where if something drifts you have to figure out what went wrong yourself.

---

## The key concepts (just enough to be useful)

There are about five concepts you actually need to understand. Everything else can wait.

**Container**: a packaged-up version of your application, including the code, dependencies, and runtime. If you've used Docker, you've built containers. Kubernetes doesn't run your code directly — it runs containers.

**Pod**: the smallest unit Kubernetes manages. A pod is one or more containers running together on the same machine, sharing networking and storage. In practice, most pods contain exactly one container. Pods are ephemeral — Kubernetes will create and destroy them freely.

**Deployment**: tells Kubernetes "I want N copies (replicas) of this pod running at all times". It handles creating pods, restarting crashed ones, and rolling out updates. This is what you'll interact with most.

**Service**: gives your pods a stable network address. Pods get created and destroyed constantly, so their IP addresses change. A Service sits in front of your pods and provides a fixed endpoint that routes traffic to whichever pods are currently alive. Think of it as a load balancer.

**Cluster**: the set of machines (called nodes) that Kubernetes manages. On GKE, Google provisions and manages the nodes for you — you just say how many and how big.

Here's how they all fit together:

![Kubernetes concepts](/assets/images/kubernetes_concepts.png)

That's genuinely all you need. Namespaces, ConfigMaps, Ingress, and the rest are useful but not essential for getting started.

---

## The walkthrough

We'll deploy a simple FastAPI app to GKE. The steps are:

1. Write a FastAPI app
2. Containerise it with Docker
3. Push the image to Google Artifact Registry
4. Write Kubernetes manifests (a Deployment and a Service)
5. Apply them to a GKE cluster

### Step 1: the FastAPI app

Nothing fancy — a prediction endpoint that takes some features and returns a score. In real life this would load a model; here we'll fake it.

```python
{% include code_snippets/kubernetes/app.py %}
```

### Step 2: Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

The `--host 0.0.0.0` is important — without it, uvicorn only listens on localhost, which means nothing outside the container can reach it.

### Step 3: build and push to Artifact Registry

First, create a repository in Artifact Registry (you only do this once):

```bash
gcloud artifacts repositories create my-repo \
    --repository-format=docker \
    --location=europe-west2 \
    --description="Docker images"
```

Then build and push:

```bash
# configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker europe-west2-docker.pkg.dev

# build the image
docker build -t europe-west2-docker.pkg.dev/YOUR_PROJECT/my-repo/fastapi-app:v1 .

# push it
docker push europe-west2-docker.pkg.dev/YOUR_PROJECT/my-repo/fastapi-app:v1
```

Replace `YOUR_PROJECT` with your GCP project ID. The tag `:v1` is arbitrary but useful — you'll reference it in the Kubernetes manifest.

### Step 4: Kubernetes manifests

This is where Kubernetes-specific stuff starts. You need two things: a Deployment (to run your pods) and a Service (to expose them).

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi-app
  template:
    metadata:
      labels:
        app: fastapi-app
    spec:
      containers:
        - name: fastapi-app
          image: europe-west2-docker.pkg.dev/YOUR_PROJECT/my-repo/fastapi-app:v1
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
```

A few things worth noting:

- `replicas: 3` means three copies of your app. Kubernetes will maintain that count.
- `resources` sets memory and CPU. `requests` is what's guaranteed; `limits` is the ceiling. The `250m` means 250 millicores — Kubernetes divides each CPU core into 1000 units of time, so 250m means your container gets 25% of one core's processing time. It's not a physical fraction of a chip, it's a time-sharing allocation enforced by the Linux kernel. These numbers matter: if you don't set them, Kubernetes can't make good scheduling decisions and your pod might get killed unexpectedly when the node runs low on memory.
- The **readinessProbe** tells Kubernetes when a pod is ready to receive traffic. It hits your `/health` endpoint — if it gets a 200, traffic starts flowing. This prevents requests hitting a pod that's still loading a model.
- The **livenessProbe** tells Kubernetes whether a pod is still alive. If it fails repeatedly, Kubernetes restarts the pod. This catches cases where your app is running but stuck.

**service.yaml:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-app-service
spec:
  type: LoadBalancer
  selector:
    app: fastapi-app
  ports:
    - port: 80
      targetPort: 8080
```

The Service watches for pods with the label `app: fastapi-app` and routes traffic to them. `type: LoadBalancer` tells GKE to provision an external IP address — so you can hit your API from the internet. Port 80 on the load balancer maps to port 8080 on the containers.

### Step 5: create a cluster and deploy

Two CLI tools are doing the work here: **`gcloud`** is Google Cloud's command-line tool for managing GCP resources (creating clusters, configuring authentication, etc.), and **`kubectl`** (pronounced "kube-control" or "kube-cuddle", depending on who you ask) is the Kubernetes command-line tool for interacting with a cluster — deploying apps, checking pod status, reading logs. You'll use `gcloud` to set things up and `kubectl` for everything after that.

```bash
# create a GKE cluster (this takes a few minutes)
gcloud container clusters create-auto my-cluster \
    --region=europe-west2

# point kubectl at your new cluster
gcloud container clusters get-credentials my-cluster \
    --region=europe-west2

# apply the manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

We're using **Autopilot mode** here, which means Google manages the nodes for you — you don't need to think about machine types or node pools. You pay per pod resource request rather than per node, which is simpler and usually cheaper for smaller workloads.

After a minute or two, check the status:

```bash
# see your pods
kubectl get pods

# see the service and its external IP
kubectl get service fastapi-app-service
```

Once the `EXTERNAL-IP` shows up (it takes a minute), you can hit your API:

```bash
curl http://EXTERNAL_IP/predict \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.5, 3.0]}'
```

That's it. Your FastAPI app is running on Kubernetes.

---

## How this actually works in practice

Everything above — building the Docker image locally, pushing it manually, running `kubectl apply` — is useful for learning, but nobody does this for real deployments. In practice, all of it is automated through a CI/CD pipeline.

The typical setup: you push code to a Git repo, and a pipeline (GitHub Actions, GitLab CI, Cloud Build, etc.) takes over. It builds the Docker image, pushes it to the registry, updates the image tag in the Kubernetes manifests, and applies them to the cluster. You never run `docker build` or `kubectl apply` yourself after the initial setup.

Here's what a minimal GitHub Actions workflow for this looks like:

```yaml
name: Deploy to GKE
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Build and push
        run: |
          gcloud auth configure-docker europe-west2-docker.pkg.dev
          docker build -t europe-west2-docker.pkg.dev/$PROJECT_ID/my-repo/fastapi-app:${{ github.sha }} .
          docker push europe-west2-docker.pkg.dev/$PROJECT_ID/my-repo/fastapi-app:${{ github.sha }}

      - name: Deploy
        run: |
          gcloud container clusters get-credentials my-cluster --region=europe-west2
          kubectl set image deployment/fastapi-app fastapi-app=europe-west2-docker.pkg.dev/$PROJECT_ID/my-repo/fastapi-app:${{ github.sha }}
```

Notice that the image tag is `${{ github.sha }}` — the Git commit hash — rather than `:v1`. This means every push produces a uniquely tagged image, so you always know exactly which code is running and can roll back to any previous commit.

Once this exists, deploying is just merging a PR. The pipeline handles the rest.

### What about Helm?

As your deployment gets more complex — multiple environments (staging, production), different configurations per environment, several microservices sharing similar YAML patterns — you'll start noticing a lot of copy-paste in your manifests. **Helm** is a templating system for Kubernetes that solves this. It lets you write your YAML once with variables (called a "chart"), and then fill in different values for each environment.

Think of it like Jinja templates for Kubernetes. Instead of maintaining separate `deployment-staging.yaml` and `deployment-prod.yaml` files, you have one template with `{{ .Values.replicas }}` and a values file per environment. Helm also handles versioning and rollbacks of your deployments.

You don't need Helm to get started — raw YAML files are fine for a single service. But if you find yourself maintaining more than a couple of manifests or deploying to multiple environments, it's worth picking up.

---

## What you get from this

At this point you might be wondering — that was a lot of YAML for something I could have deployed to a single VM with `uvicorn app:app`. Fair. Here's what Kubernetes gives you that a VM doesn't:

- **Self-healing**: if a pod crashes, Kubernetes restarts it automatically. If a node goes down, your pods get rescheduled to healthy nodes. 

- **Scaling**: change `replicas: 3` to `replicas: 10` and apply the file. Or better, set up a HorizontalPodAutoscaler that adds and removes pods based on CPU usage or request volume. 

- **Rolling updates**: push a new image tag and Kubernetes swaps out pods one at a time. At no point is the service fully down. If the new version crashes, it stops the rollout automatically.

- **Resource isolation**: each pod has defined resource limits, so one misbehaving service can't consume all the memory on a machine and take down everything else.

- **Reproducibility**: the entire deployment is defined in version-controlled YAML files. Anyone on the team can see exactly what's running and recreate it.

---

## When you don't need Kubernetes

Kubernetes is great but it's not always the right tool. Some honest takes:

- **If you're serving a single model and don't need to scale beyond one machine**, a managed service like Cloud Run or even just running it on a VM is simpler and cheaper. Cloud Run gives you auto-scaling and HTTPS with almost no configuration. And if you need async work — queueing up batch predictions, retrying failed calls, rate-limiting downstream services — **Cloud Tasks** pairs naturally with Cloud Run. You get a managed task queue that calls your Cloud Run endpoint with configurable retries and rate limits, no Kubernetes needed. For a lot of data science workloads (model API + some background processing), Cloud Run + Cloud Tasks covers it.

- **If you're prototyping**, Kubernetes adds overhead that slows you down. Get the model working first, then think about infrastructure.

- **If you don't have a team that can support it**, Kubernetes introduces operational complexity. Autopilot reduces this a lot, but you'll still hit debugging situations where things fail in ways that aren't obvious — a service can't talk to another service, a pod can't find an endpoint, a container behaves differently than it did on your laptop.

A reasonable rule of thumb: if you have multiple services that need to talk to each other, or you need fine-grained control over scaling, resource allocation, and deployment strategy, Kubernetes earns its complexity. For a single API endpoint, start with Cloud Run and graduate to Kubernetes when you outgrow it.

---

## Common things that trip you up

A few things I wish someone had told me earlier:

- **Image pull errors**: if your pods show `ImagePullBackOff`, it almost always means Kubernetes can't access your container image. Either the image path is wrong, the tag doesn't exist, or the cluster doesn't have permission to pull from your registry. On GKE with Artifact Registry in the same project, permissions are usually set up automatically — but if you're pulling from a different project, you'll need to grant access.

- **CrashLoopBackOff**: your container is starting and immediately crashing, and Kubernetes keeps restarting it (with increasing delays). Check the logs with `kubectl logs <pod-name>`. It's usually a missing environment variable, a failed import, or the app binding to the wrong port.

- **Pending pods**: if a pod stays in `Pending` state, Kubernetes can't find a node with enough resources. Either your resource requests are too high or you need more/bigger nodes. On Autopilot, GKE will provision nodes automatically, but it takes a couple of minutes.

- **Forgetting `--host 0.0.0.0`**: your app works locally in Docker but gets no traffic in Kubernetes. The container is only listening on localhost, and the Service can't reach it. Uvicorn, Gunicorn, and Flask all default to localhost — you need to explicitly bind to `0.0.0.0`.

---

## Further reading

- **Kubernetes docs** — [Overview](https://kubernetes.io/docs/concepts/overview/): the official concepts page is genuinely well-written and worth reading once you have the basics down
- **Google Cloud** — [GKE Quickstart](https://cloud.google.com/kubernetes-engine/docs/deploy-app-cluster): Google's own walkthrough for deploying to GKE, more detailed on the GCP-specific parts
- **Julia Evans** — [A few things I've learned about Kubernetes](https://jvns.ca/blog/2017/06/04/learning-about-kubernetes/): this one is good for demystifying the operational side
- **FastAPI docs** — [Deployment](https://fastapi.tiangolo.com/deployment/): covers Docker and cloud deployment options for FastAPI specifically
- **k9s** — [k9scli.io](https://k9scli.io/): a terminal UI that wraps `kubectl` — lets you browse pods, read logs, and debug interactively instead of memorising commands. Highly recommended once you're past the initial setup
- **Chip Huyen (2022)** — [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/): Chapter 7 covers deployment and infrastructure for ML, including containers and orchestration, aimed at ML engineers
