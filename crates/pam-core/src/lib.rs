//! The core of Pam: everything except drawing.
//!
//! Front-ends start a [`Meeting`], draw the [`CoreEvent`]s it sends, and stop it.

pub mod agent;
pub use pam_audio as audio;
pub(crate) mod brief;
pub mod clock;
pub mod config;
pub mod context;
pub mod event;
pub mod folder;
pub mod identity;
pub(crate) mod intro;
pub(crate) mod invocation;
pub mod llm;
pub(crate) mod local;
pub mod meeting;
pub mod models;
pub mod notes;
pub(crate) mod pipeline;
pub mod question;
pub mod record;
pub(crate) mod rpc;
pub(crate) mod scribe;
pub mod task;
pub mod usage;

pub use agent::AgentConfig;
pub use config::Config;
pub use event::{Command, CoreEvent, EventId, Line, PreviewLine, SpkId, TaskId, UndoId, UtteranceId};
pub use identity::{Overlay, Speakers};
pub use meeting::{Input, Llm, Meeting, MeetingConfig, StopReport};
pub use models::{AgentInfo, IntentInfo, ModelInfo, ModelStatus, ScribeInfo, Source};
pub use notes::Notes;
pub use pam_speech::Ts;
pub use task::{Target, TaskResult, TaskState, TaskStatus, Via};
pub use usage::Usage;
