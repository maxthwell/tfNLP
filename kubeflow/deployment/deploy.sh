export PATH=${PATH}:$(pwd)
export KFAPP="myapp"

# Specify credentials for the default user.
export KUBEFLOW_USER_EMAIL="admin@kubeflow.org"
export KUBEFLOW_PASSWORD="12341234"

mkdir -p /tmp/kfapp/myapp
cp app.yaml /tmp/kfapp/myapp/app.yaml
unzip kubeflow/kubeflow-0.6.2.zip -d /tmp/kfapp
unzip manifests/manifests-0.6.2.zip -d /tmp/kfapp
cd /tmp/kfapp/myapp
kfctl generate all -V
kfctl apply all -V
