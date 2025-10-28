use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use candle_core::{Device, DType, Tensor, D, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, VarMap, SGD, Optimizer};
use candle_nn::{ops, loss};
use candle_datasets::vision::mnist;
use tokio::net::TcpListener;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TensorPayload { dims: Vec<usize>, values: Vec<f32> }
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelPayload { weight: TensorPayload, bias: TensorPayload }

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct TrainEnvelope {
    model: String,
    shard_start: usize,
    shard_len: usize,
    weights: ModelPayload,
}
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct TrainReturn {
    weights: ModelPayload,
    train_loss: f32,
    test_acc: f32,
}

#[derive(Clone)]
struct AppState {
    device: Device,
    dataset: Arc<candle_datasets::vision::Dataset>,
}

fn payload_to_tensor(p: &TensorPayload, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(p.values.clone(), p.dims.clone(), device)?)
}
fn tensor_to_payload(t: &Tensor) -> Result<TensorPayload> {
    Ok(TensorPayload { dims: t.dims().to_vec(), values: t.flatten_all()?.to_vec1::<f32>()? })
}

#[tokio::main]
async fn main() -> Result<()> {
    // args: --port 3001 --server http://127.0.0.1:3000
    let port: u16 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3001);
    let server = std::env::args().nth(4).unwrap_or_else(|| "http://127.0.0.1:3000".into());

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dataset = Arc::new(mnist::load()?);
    let st = AppState { device, dataset };

    // Join (register) automatically at startup
    let http = reqwest::Client::new();
    let _ = http.post(format!("{server}/register"))
        .json(&serde_json::json!({"client_addr": format!("http://127.0.0.1:{port}"), "model": "mnist-linear"}))
        .send().await;

    let app = Router::new()
        .route("/train", post(train))
        .with_state(st);

    let addr: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    let listener = TcpListener::bind(addr).await?;
    println!("Client on {}, joined server {server}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn train(State(st): State<AppState>, Json(env): Json<TrainEnvelope>) -> Result<Json<TrainReturn>, axum::http::StatusCode> {
    let TrainEnvelope { shard_start, shard_len, weights, .. } = env;

    // Rebuild linear with provided global weights
    let w = payload_to_tensor(&weights.weight, &st.device).map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;
    let b = payload_to_tensor(&weights.bias, &st.device).map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;
    let lin = Linear::new(w, Some(b));

    // Shard slice
    let x = st.dataset.train_images.i(shard_start..(shard_start+shard_len)).map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;
    let y = st.dataset.train_labels.i(shard_start..(shard_start+shard_len)).map_err(|_| axum::http::StatusCode::BAD_REQUEST)?
        .to_dtype(DType::U32).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;

    // One short training step with autograd
    let vm = VarMap::new();
    let _vs = VarBuilder::from_varmap(&vm, DType::F32, &st.device);
    let mut sgd = SGD::new(vm.all_vars(), 1.0f64).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let logits = lin.forward(&x).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let log_sm = ops::log_softmax(&logits, D::Minus1).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let loss = loss::nll(&log_sm, &y).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    sgd.backward_step(&loss).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;

    let test_logits = lin.forward(&x).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let ok = test_logits.argmax(D::Minus1).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .eq(&y).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .to_dtype(DType::F32).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .sum_all().map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .to_scalar::<f32>().map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let acc = ok / (shard_len as f32);

    let ret = TrainReturn {
        weights: ModelPayload {
            weight: tensor_to_payload(lin.weight()).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?,
            bias: tensor_to_payload(lin.bias().unwrap()).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?,
        },
        train_loss: loss.to_scalar::<f32>().map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?,
        test_acc: acc,
    };
    Ok(Json(ret))
}
