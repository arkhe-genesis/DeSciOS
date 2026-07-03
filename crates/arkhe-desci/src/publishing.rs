//! Publicação descentralizada de datasets científicos

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{DesciError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub description: String,
    pub format: String,
    pub version: String,
    pub author_did: String,
    pub license: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub checksum_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsPublishResult {
    pub cid: String,
    pub gateway_url: String,
    pub size_bytes: u64,
}

pub struct IpfsClient {
    api_url: String,
    gateway_url: String,
    http_client: reqwest::Client,
}

impl IpfsClient {
    pub fn local() -> Self {
        Self {
            api_url: "http://127.0.0.1:5001/api/v0".to_string(),
            gateway_url: "http://127.0.0.1:8080/ipfs".to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    pub fn new(api_url: &str, gateway_url: &str) -> Self {
        Self {
            api_url: api_url.to_string(),
            gateway_url: gateway_url.to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    #[cfg(feature = "ipfs")]
    pub async fn add_file(&self, path: &str) -> Result<IpfsPublishResult> {
        let file_bytes = tokio::fs::read(path).await
            .map_err(|e| DesciError::Io(e))?;

        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(file_bytes.clone())
                .file_name(std::path::Path::new(path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("data")
                    .to_string()));

        let response = self.http_client
            .post(format!("{}/add", self.api_url))
            .multipart(form)
            .send()
            .await
            .map_err(|e| DesciError::IpfsError(format!("Request failed: {}", e)))?
            .error_for_status()
            .map_err(|e| DesciError::IpfsError(format!("API error: {}", e)))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| DesciError::IpfsError(format!("JSON parse error: {}", e)))?;

        let cid = response["Hash"]
            .as_str()
            .ok_or_else(|| DesciError::IpfsError("No CID in response".to_string()))?
            .to_string();

        let size = response["Size"]
            .as_u64()
            .unwrap_or(file_bytes.len() as u64);

        info!(cid = %cid, size = size, "File added to IPFS");

        Ok(IpfsPublishResult {
            cid: cid.clone(),
            gateway_url: format!("{}/{}", self.gateway_url, cid),
            size_bytes: size,
        })
    }

    #[cfg(feature = "ipfs")]
    pub async fn add_bytes(&self, data: &[u8], filename: &str) -> Result<IpfsPublishResult> {
        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(data.to_vec())
                .file_name(filename.to_string()));

        let response = self.http_client
            .post(format!("{}/add", self.api_url))
            .multipart(form)
            .send()
            .await
            .map_err(|e| DesciError::IpfsError(format!("Request failed: {}", e)))?
            .error_for_status()
            .map_err(|e| DesciError::IpfsError(format!("API error: {}", e)))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| DesciError::IpfsError(format!("JSON parse error: {}", e)))?;

        let cid = response["Hash"]
            .as_str()
            .ok_or_else(|| DesciError::IpfsError("No CID in response".to_string()))?
            .to_string();

        let size = response["Size"]
            .as_u64()
            .unwrap_or(data.len() as u64);

        Ok(IpfsPublishResult {
            cid: cid.clone(),
            gateway_url: format!("{}/{}", self.gateway_url, cid),
            size_bytes: size,
        })
    }

    pub fn api_url(&self) -> &str {
        &self.api_url
    }

    pub fn gateway_url(&self) -> &str {
        &self.gateway_url
    }
}

pub struct WormGraphNotifier {
    endpoint: String,
}

impl WormGraphNotifier {
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
        }
    }

    pub async fn notify_publication(
        &self,
        cid: &str,
        metadata: &DatasetMetadata,
    ) -> Result<String> {
        let notification_id = blake3::hash(
            format!("{}:{}:{}", cid, metadata.name, chrono::Utc::now().timestamp_millis()).as_bytes()
        ).to_string();

        info!(
            notification_id = %notification_id,
            cid = %cid,
            dataset = %metadata.name,
            endpoint = %self.endpoint,
            "WormGraph notification sent (stub)"
        );

        Ok(notification_id)
    }
}

pub struct DeSciPublisher {
    ipfs_client: IpfsClient,
    wormgraph: WormGraphNotifier,
}

impl DeSciPublisher {
    pub fn local() -> Self {
        Self {
            ipfs_client: IpfsClient::local(),
            wormgraph: WormGraphNotifier::new("http://localhost:50051"),
        }
    }

    pub fn new(ipfs_api: &str, ipfs_gateway: &str, wormgraph_endpoint: &str) -> Self {
        Self {
            ipfs_client: IpfsClient::new(ipfs_api, ipfs_gateway),
            wormgraph: WormGraphNotifier::new(wormgraph_endpoint),
        }
    }

    #[cfg(feature = "ipfs")]
    pub async fn publish(
        &self,
        file_path: &str,
        metadata: DatasetMetadata,
    ) -> Result<PublishResult> {
        let ipfs_result = self.ipfs_client.add_file(file_path).await?;
        let notification_id = self.wormgraph
            .notify_publication(&ipfs_result.cid, &metadata)
            .await?;

        info!(
            cid = %ipfs_result.cid,
            notification = %notification_id,
            "Dataset published successfully"
        );

        Ok(PublishResult {
            cid: ipfs_result.cid,
            gateway_url: ipfs_result.gateway_url,
            size_bytes: ipfs_result.size_bytes,
            notification_id,
            metadata,
        })
    }

    #[cfg(feature = "ipfs")]
    pub async fn publish_bytes(
        &self,
        data: &[u8],
        filename: &str,
        metadata: DatasetMetadata,
    ) -> Result<PublishResult> {
        let ipfs_result = self.ipfs_client.add_bytes(data, filename).await?;
        let notification_id = self.wormgraph
            .notify_publication(&ipfs_result.cid, &metadata)
            .await?;

        Ok(PublishResult {
            cid: ipfs_result.cid,
            gateway_url: ipfs_result.gateway_url,
            size_bytes: ipfs_result.size_bytes,
            notification_id,
            metadata,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResult {
    pub cid: String,
    pub gateway_url: String,
    pub size_bytes: u64,
    pub notification_id: String,
    pub metadata: DatasetMetadata,
}

#[allow(dead_code)]
pub struct CcipClient {
    _router_address: String,
    _chain_id: u64,
}

#[allow(dead_code)]
impl CcipClient {
    pub fn new(router_address: &str, chain_id: u64) -> Self {
        Self {
            _router_address: router_address.to_string(),
            _chain_id: chain_id,
        }
    }

    pub async fn send_message(&self, _payload: &[u8]) -> Result<String> {
        Err(DesciError::NotImplemented(
            "CCIP integration requires ethers-rs/alloy + smart contract deployment. \
             See ADR-026 Section 4: use WormGraph gRPC for internal ARKHE notifications.".to_string()
        ))
    }
}
