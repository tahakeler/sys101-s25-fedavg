use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, routing::{get, post}, Json, Router};
use candle_core::{Device, DType, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use candle_datasets::vision::mnist;
use futures::future::join_all;
use reqwest::Client as Http;
use tokio::{net::TcpListener, sync::RwLock};
use thiserror::Error;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TensorPayload { dims: Vec<usize>, values: Vec<f32> }
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelPayload { weight: TensorPayload, bias: TensorPayload }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RegisterReq { client_addr: String, model: String }
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct InitReq { model: String }
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrainReq { model: String, rounds: usize }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GetResp {
    status: String,
    accuracy: Option<f32>,
    model: Option<ModelPayload>,
}

#[derive(Debug, Clone)]
struct GlobalModel {
    weight: Tensor,
    bias: Tensor,
    status: String,
    accuracy: Option<f32>,
}

#[derive(Clone)]
struct AppState {
    http: Http,
    device: Device,
    dataset: Arc<candle_datasets::vision::Dataset>,
    models: Arc<RwLock<HashMap<String, GlobalModel>>>,
    clients: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    shards: Arc<RwLock<HashMap<String, Vec<(usize, usize)>>>>,
}

#[derive(Error, Debug)]
enum ServerError {
    #[error("model not found")]
    ModelNotFound,
    #[error("no clients registered")]
    NoClients,
}

fn tensor_to_payload(t: &Tensor) -> Result<TensorPayload> {
    Ok(TensorPayload {
        dims: t.dims().to_vec(),
        values: t.flatten_all()?.to_vec1::<f32>()?,
    })
}
fn payload_to_tensor(p: &TensorPayload, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(p.values.clone(), p.dims.clone(), device)?)
}

fn fresh_linear(vs: VarBuilder) -> Result<Linear> {
    let w = vs.get_with_hints((10, 784), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
    let b = vs.get_with_hints(10, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(w, Some(b)))
}

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

#[tokio::main]
async fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dataset = Arc::new(mnist::load()?);

    let state = AppState {
        http: Http::new(),
        device,
        dataset,
        models: Arc::new(RwLock::new(HashMap::new())),
        clients: Arc::new(RwLock::new(HashMap::new())),
        shards: Arc::new(RwLock::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/register", post(register))
        .route("/init", post(init_model))
        .route("/train", post(train_rounds))
        .route("/get", get(get_model))
        .route("/test", get(test_model))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:3000".parse().unwrap();
    let listener = TcpListener::bind(addr).await?;
    println!("Server on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn register(State(st): State<AppState>, Json(req): Json<RegisterReq>) -> Json<String> {
    let mut c = st.clients.write().await;
    c.entry(req.model.clone()).or_default().insert(req.client_addr.clone());
    Json("ok".into())
}

async fn init_model(State(st): State<AppState>, Json(req): Json<InitReq>) -> Result<Json<String>, axum::http::StatusCode> {
    let vm = candle_nn::VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &st.device);
    let m = fresh_linear(vs).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let weight = m.weight().clone();
    let bias = m.bias().unwrap().clone();
    let mut models = st.models.write().await;
    models.insert(req.model.clone(), GlobalModel { weight, bias, status: "initialized".into(), accuracy: None });

    // IID shards
    let n = st.dataset.train_images.dims()[0];
    let clients = st.clients.read().await;
    let Some(set) = clients.get(&req.model) else {
        return Err(axum::http::StatusCode::BAD_REQUEST);
    };
    let kc = set.len().max(1);
    let chunk = n / kc;
    let mut shards = vec![];
    let mut start = 0usize;
    for i in 0..kc {
        let len = if i == kc-1 { n - start } else { chunk };
        shards.push((start, len));
        start += len;
    }
    st.shards.write().await.insert(req.model.clone(), shards);

    Ok(Json("ok".into()))
}

async fn train_rounds(State(st): State<AppState>, Json(req): Json<TrainReq>) -> Result<Json<String>, axum::http::StatusCode> {
    let mut models = st.models.write().await;
    let Some(global) = models.get_mut(&req.model) else { return Err(axum::http::StatusCode::NOT_FOUND); };
    global.status = "training".into();
    drop(models);

    let clients = st.clients.read().await;
    let Some(addrs) = clients.get(&req.model) else { return Err(axum::http::StatusCode::BAD_REQUEST); };
    let shards = st.shards.read().await;
    let Some(shards_vec) = shards.get(&req.model) else { return Err(axum::http::StatusCode::BAD_REQUEST); };

    for round in 0..req.rounds {
        println!("=== Round {round} ===");
        let weight_pl = {
            let m = st.models.read().await;
            tensor_to_payload(&m.get(&req.model).unwrap().weight).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        };
        let bias_pl = {
            let m = st.models.read().await;
            tensor_to_payload(&m.get(&req.model).unwrap().bias).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        };

        let http = st.http.clone();
        let futs = addrs.iter().cloned().zip(shards_vec.iter().cloned()).map(|(addr,(start,len))| {
            let http = http.clone();
            let env = TrainEnvelope {
                model: req.model.clone(),
                shard_start: start,
                shard_len: len,
                weights: ModelPayload { weight: weight_pl.clone(), bias: bias_pl.clone() }
            };
            async move {
                let res = http.post(format!("{addr}/train")).json(&env).send().await?;
                res.error_for_status_ref()?;
                let ret = res.json::<TrainReturn>().await?;
                Ok::<TrainReturn, reqwest::Error>(ret)
            }
        });

        let mut results = Vec::new();
        for r in join_all(futs).await {
            if let Ok(x) = r { results.push(x) }
        }
        if results.is_empty() { return Err(axum::http::StatusCode::BAD_GATEWAY); }

        // FedAvg with proper scalar multiply
        let mut agg_w = payload_to_tensor(&results[0].weights.weight, &st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
        let mut agg_b = payload_to_tensor(&results[0].weights.bias, &st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
        for ret in results.iter().skip(1) {
            let w = payload_to_tensor(&ret.weights.weight, &st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
            let b = payload_to_tensor(&ret.weights.bias, &st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
            agg_w = (&agg_w + &w).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
            agg_b = (&agg_b + &b).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        let scale = 1f32 / (results.len() as f32);
        agg_w = agg_w.mul(&Tensor::new(scale, &st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
        agg_b = agg_b.mul(&Tensor::new(scale, &st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;

        let mut m = st.models.write().await;
        if let Some(g) = m.get_mut(&req.model) {
            g.weight = agg_w;
            g.bias = agg_b;
        }
    }

    let mut m = st.models.write().await;
    if let Some(g) = m.get_mut(&req.model) {
        g.status = "ready".into();
    }
    Ok(Json("ok".into()))
}

async fn get_model(State(st): State<AppState>) -> Json<GetResp> {
    let m = st.models.read().await;
    if let Some(g) = m.get("mnist-linear") {
        let w = tensor_to_payload(&g.weight).ok();
        let b = tensor_to_payload(&g.bias).ok();
        return Json(GetResp {
            status: g.status.clone(),
            accuracy: g.accuracy,
            model: match (w, b) {
                (Some(weight), Some(bias)) => Some(ModelPayload { weight, bias }),
                _ => None
            }
        })
    }
    Json(GetResp { status: "unknown".into(), accuracy: None, model: None })
}

async fn test_model(State(st): State<AppState>) -> Result<Json<GetResp>, axum::http::StatusCode> {
    let m = st.models.read().await;
    let Some(g) = m.get("mnist-linear") else { return Err(axum::http::StatusCode::NOT_FOUND); };

    let lin = Linear::new(g.weight.clone(), Some(g.bias.clone()));
    let test_images = st.dataset.test_images.to_device(&st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let logits = lin.forward(&test_images).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let sum_ok = logits.argmax(D::Minus1).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .eq(&st.dataset.test_labels.to_device(&st.device).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .to_dtype(DType::F32).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .sum_all().map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
        .to_scalar::<f32>().map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let acc = sum_ok / (st.dataset.test_labels.dims()[0] as f32);

    Ok(Json(GetResp { status: "ready".into(), accuracy: Some(acc), model: None }))
}
