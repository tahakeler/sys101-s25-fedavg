use anyhow::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use candle_core::{Device, DType, Tensor, IndexOp, D};

use candle_nn::{ops, encoding, loss};
use candle_datasets::vision::mnist;

/// Model geometry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelShape {
    pub in_dim: usize,
    pub out_dim: usize,
}
pub fn mnist_shape() -> ModelShape { ModelShape { in_dim: 28 * 28, out_dim: 10 } }

/// Flat parameters for a single linear layer (W,b)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearParams {
    /// Flattened row-major [out_dim, in_dim]
    pub w: Vec<f32>,
    /// [out_dim]
    pub b: Vec<f32>,
    pub shape: ModelShape,
}

impl LinearParams {
    pub fn zeros(shape: ModelShape) -> Self {
        let w = vec![0.0; shape.out_dim * shape.in_dim];
        let b = vec![0.0; shape.out_dim];
        Self { w, b, shape }
    }

    pub fn average(params: &[Self]) -> Self {
        assert!(!params.is_empty());
        let shape = params[0].shape.clone();
        let mut out = Self::zeros(shape.clone());
        let n = params.len() as f32;
        for (i, v) in out.w.iter_mut().enumerate() {
            *v = params.iter().map(|p| p.w[i]).sum::<f32>() / n;
        }
        for (i, v) in out.b.iter_mut().enumerate() {
            *v = params.iter().map(|p| p.b[i]).sum::<f32>() / n;
        }
        out
    }

    pub fn to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let w = Tensor::from_vec(
            self.w.clone(),
            (self.shape.out_dim, self.shape.in_dim),
            device,
        )?;
        let b = Tensor::from_vec(self.b.clone(), self.shape.out_dim, device)?;
        Ok((w, b))
    }

    pub fn from_tensors(w: &Tensor, b: &Tensor) -> Result<Self> {
        let (out_dim, in_dim) = w.dims2()?;
        let shape = ModelShape { in_dim, out_dim };
        let wv: Vec<f32> = w.flatten_all()?.to_vec1()?;
        let bv: Vec<f32> = b.flatten_all()?.to_vec1()?;
        Ok(Self { w: wv, b: bv, shape })
    }
}

/// Training hyperparameters for local client steps
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,          // learning rate
    pub sample_ratio: f32 // fraction of local data to use (0,1]
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self { epochs: 1, batch_size: 64, lr: 0.1, sample_ratio: 1.0 }
    }
}

/// Metrics returned from a local training step
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainMetrics {
    pub loss: f32,
    pub acc: f32,
}

/// Convenience holder for MNIST tensors (already on target device)
pub struct LocalDataset {
    pub device: Device,
    pub train_images: Tensor, // [N, 784] f32
    pub train_labels: Tensor, // [N] u32
    pub test_images: Tensor,  // [M, 784] f32
    pub test_labels: Tensor,  // [M] u32
    n_train: usize,
    n_test: usize,
}

impl LocalDataset {
    /// Load full MNIST tensors using Candle's datasets API.
    pub fn load_mnist(_train: bool, device: &Device) -> Result<Self> {
        let ds = mnist::load()?;
        let train_images = ds.train_images.to_device(device)?; // [N, 784] f32
        let train_labels = ds.train_labels.to_device(device)?.to_dtype(DType::U32)?;
        let test_images = ds.test_images.to_device(device)?;
        let test_labels = ds.test_labels.to_device(device)?.to_dtype(DType::U32)?;
        let n_train = train_images.dims()[0];
        let n_test = test_images.dims()[0];
        Ok(Self {
            device: device.clone(),
            train_images, train_labels, test_images, test_labels,
            n_train, n_test
        })
    }

    /// IID sample a subset from local train set by ratio (shuffle then take top-k)
    pub fn iid_sample(&self, ratio: f32, rng: &mut impl Rng) -> Result<(Tensor, Tensor)> {
        let take = ((self.n_train as f32) * ratio)
            .clamp(1.0, self.n_train as f32) as usize;
        let mut idxs: Vec<usize> = (0..self.n_train).collect();
        idxs.shuffle(rng);
        idxs.truncate(take);
        let idxs_t = Tensor::from_vec(
            idxs.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            take,
            &self.device,
        )?;
        let x = self.train_images.index_select(&idxs_t, 0)?;
        let y = self.train_labels.index_select(&idxs_t, 0)?;
        Ok((x, y))
    }
}

/// Train a linear model **without autograd** using manual gradients for
/// softmax-cross-entropy. This avoids optimizer API shifts and keeps
/// compatibility with Candle-from-GitHub.
pub fn train_local_linear(
    initial: &LinearParams,
    config: &TrainConfig,
    dataset: &LocalDataset,
    rng: &mut impl Rng,
) -> Result<(LinearParams, TrainMetrics)> {
    let device = &dataset.device;
    let (mut w, mut b) = initial.to_tensors(device)?; // w:[10,784], b:[10]
    let (x_all, y_all) = dataset.iid_sample(config.sample_ratio, rng)?; // x:[N,784], y:[N]u32

    let n = x_all.dims2()?.0;
    let bs = config.batch_size.max(1);
    let steps = (n + bs - 1) / bs;

    let mut last_loss = 0f32;

    for _ in 0..config.epochs {
        // shuffle indices
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(rng);
        let order_t = Tensor::from_vec(
            order.iter().map(|&i| i as i64).collect::<Vec<_>>(),
            n,
            device,
        )?;
        let x_shuf = x_all.index_select(&order_t, 0)?;
        let y_shuf = y_all.index_select(&order_t, 0)?;

        for s in 0..steps {
            let st = s * bs;
            let en = (st + bs).min(n);
            let xb = x_shuf.i(st..en)?;        // [B,784]
            let yb = y_shuf.i(st..en)?;        // [B]u32
            let bsz = en - st;

            // logits = xb @ w^T + b
            let logits = xb.matmul(&w.t()?)?; // [B,10]
            let logits = logits.broadcast_add(&b.unsqueeze(0)?)?; // broadcast bias

            // softmax and log-softmax
            let probs = ops::softmax(&logits, D::Minus1)?;    // [B,10]
            let log_sm = ops::log_softmax(&logits, D::Minus1)?; // [B,10]

            // one-hot labels
            let y_u32 = yb.to_dtype(DType::U32)?;
            let y_one = encoding::one_hot(y_u32, initial.shape.out_dim, 1.0f32, 0.0f32)?; // [B,10], f32

            // NLL loss using built-in function
            let loss = loss::nll(&log_sm, &yb)?;                    // scalar
            last_loss = loss.to_scalar::<f32>()?;

            // Gradients:
            // dL/dlogits = probs - y_one
            let diff = (&probs - &y_one)?;                          // [B,10]
            // dL/dW = (diff^T @ xb) / B
            let grad_w = diff.t()?.matmul(&xb)?.div(&Tensor::new(bsz as f32, device)?)?;                     // [10,784]
            // dL/db = mean(diff, dim=0)
            let grad_b = diff.mean(0)?;                             // [10]

            // SGD update
            let step_w = grad_w.mul(&Tensor::new(config.lr, device)?)?;             // [10,784]
            let step_b = grad_b.mul(&Tensor::new(config.lr, device)?)?;             // [10]
            w = w.sub(&step_w)?;                                    // [10,784]
            b = b.sub(&step_b)?;                                    // [10]
        }
    }

    // Accuracy on sampled data
    let logits = x_all.matmul(&w.t()?)?.broadcast_add(&b.unsqueeze(0)?)?;
    let preds = logits.argmax(D::Minus1)?;
    let correct = preds.eq(&y_all)?
        .to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    let acc = correct / (n as f32);

    let updated = LinearParams::from_tensors(&w, &b)?;
    Ok((updated, TrainMetrics { loss: last_loss, acc }))
}

/// Evaluate accuracy on the dataset's full test split
pub fn test_linear(params: &LinearParams, dataset: &LocalDataset) -> Result<f32> {
    let device = &dataset.device;
    let (w, b) = params.to_tensors(device)?;
    let logits = dataset
        .test_images
        .matmul(&w.t()?)?
        .broadcast_add(&b.unsqueeze(0)?)?;
    let preds = logits.argmax(D::Minus1)?;
    let correct = preds
        .eq(&dataset.test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(correct / (dataset.n_test as f32))
}
