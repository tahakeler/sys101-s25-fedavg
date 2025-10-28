use rand::{rngs::StdRng, SeedableRng};

pub fn seeded(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

pub fn parse_addr(default_port: u16) -> String {
    std::env::var("ADDR").unwrap_or_else(|_| format!("0.0.0.0:{}", default_port))
}

pub fn parse_url(default: &str) -> String {
    std::env::var("URL").unwrap_or_else(|_| default.to_string())
}
