//! `pam`: runs a meeting with the terminal dashboard (`pam listen`), and lets
//! the agent read it (`pam status`, `pam transcript`, …).

mod cli;
mod dashboard;
mod listen;
mod models;
mod note;
mod read;
mod say;
mod task;

use std::process::ExitCode;

fn main() -> ExitCode {
    cli::run(std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect())
}
