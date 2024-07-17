# ray-lightning-demo
This project sets up a simple project in which I showcase the integration between Ray and PyTorch Lightning. An image classifier is trained using Ray Train and Lightning, in a Ray cluster running on AWS EC2 instances.

# Job Submission

```bash
ray job submit \
    --runtime-env-json='{"working_dir": ".", "pip": ["torch==2.2.2", "torchvision==0.17.2"]}' \
    -- python ray_lightning_demo/batch_classification.py
```


ray job submit --working-dir . -- python ray_lightning_demo/main.py