use serde::{Deserialize, Serialize};

use crate::model::{LinearParams, TrainConfig, TrainMetrics, ModelShape};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub client_url: String,
    pub model: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub ok: bool,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitRequest {
    pub model: String,
    pub shape: Option<ModelShape>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainRoundRequest {
    pub model: String,
    pub rounds: usize,
    pub clients_per_round: usize,
    pub train: TrainConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetModelResponse {
    pub status: String, // "uninitialized" | "training" | "ready"
    pub params: Option<LinearParams>,
    pub round: usize,
    pub registered_clients: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientTrainRequest {
    pub model: String,
    pub params: LinearParams,
    pub train: TrainConfig,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientTrainResponse {
    pub params: LinearParams,
    pub metrics: TrainMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestRequest {
    pub model: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestResponse {
    pub acc: f32,
}
