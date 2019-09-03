# kubeflow部署指南

## 首先需要部署proton的k8s环境：

部署指南：<http://confluence.eisoo.com/pages/viewpage.action?pageId=54461065>

## 创建一个nfs服务，将用于k8s的PV的后备存储

```sh
yum -y install nfs
for i in $(seq 10);do mkdir -p /nfs/$i; done
systemctl restart rpcbind
systemctl restart nfs
```

## 创建PV，PVC

将nfs-volume.yaml中的nfs的IP地址改成真实的nfs服务的ip地址，然后执行：

```sh
kubectl create -f nfs-volume.yaml
```

## 部署kubeflow：

```
sh deployment.sh
```

