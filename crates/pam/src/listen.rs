//! `pam listen`: start a meeting, draw it until `q`, then stop it.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};
use pam_core::clock::mmss;
use pam_core::folder::meetings_root;
use pam_core::models::{self, Choice, Pick, Use};
use pam_core::{Config, Meeting, MeetingConfig};
use pam_speech::store::Store;
use ratatui::crossterm::event::{self, Event, KeyEventKind};

use crate::dashboard::{Action, Dashboard};

pub struct Options {
    pub input: PathBuf,
    pub realtime: bool,
    pub title: String,
    /// The assistant's name; `None` for the config file's, else Pam.
    pub name: Option<String>,
    pub attendees: Vec<String>,
    /// `--scribe`: the scribe's model, or `off`; `None` for the config file's, else detected.
    pub scribe: Option<String>,
    /// `--intent`: the intent check's model the same way, or `local`; `None`
    /// for the config file's, else the scribe's.
    pub intent: Option<String>,
    /// `--agent`: the agent's model the same way.
    pub agent: Option<String>,
}

/// How often the dashboard redraws while nothing happens (the wave animates).
const FRAME: Duration = Duration::from_millis(80);

pub fn run(opts: Options) -> Result<()> {
    let input = opts.input.display().to_string();
    let config = Config::load()?;
    let mut cfg = if input == "mic" {
        MeetingConfig::mic(meetings_root()?, &config)
    } else if input.ends_with(".wav") {
        MeetingConfig::wav(meetings_root()?, opts.input, &config)
    } else {
        MeetingConfig::fake(meetings_root()?, opts.input)?
    };
    cfg.realtime = opts.realtime;
    cfg.title = opts.title;
    let name = opts.name.or(config.name.clone()).unwrap_or_else(|| "Pam".into());
    if name.trim().is_empty() {
        bail!("the assistant's name is empty: give --name a name, or set `name` in the config file");
    }
    cfg.assistant_name = name.trim().to_string();
    if !(1..=pam_core::question::MAX_TIMEOUT).contains(&config.question_timeout) {
        bail!(
            "`question_timeout` in the config file must be 1 to {} seconds, not {}",
            pam_core::question::MAX_TIMEOUT,
            config.question_timeout
        );
    }
    cfg.question_timeout = config.question_timeout;
    cfg.attendees = opts.attendees;
    let scribe = Pick::new(Use::Scribe, opts.scribe.as_deref(), config.scribe_model.as_deref());
    let intent = Pick::new(Use::Intent, opts.intent.as_deref(), config.intent_model.as_deref());
    let agent = Pick::new(Use::Agent, opts.agent.as_deref(), config.agent_model.as_deref());
    let choice = Choice {
        pi: config.pi.clone(),
        settings: models::pi_settings(),
        scribe: scribe.map_err(anyhow::Error::msg)?,
        intent: intent.map_err(anyhow::Error::msg)?,
        agent: agent.map_err(anyhow::Error::msg)?,
    };
    // Nothing is downloaded here, where gigabytes would read as a hang: a
    // missing model stops it, saying what to pull.
    models::check_stored(&Store::open()?, &cfg.asr, &cfg.diarizer, choice.loads_local())?;
    eprintln!("Checking pi and the models…");
    // The agent runs this `pam`, installed or not.
    cfg.use_models(&choice, std::env::current_exe().ok());
    let notes = cfg.scribe.is_some();
    let title = if cfg.title.is_empty() { "Meeting".to_string() } else { cfg.title.clone() };

    if !cfg.asr.starts_with("fake:") {
        eprintln!("Loading the speech models ({}, {})…", cfg.asr, cfg.diarizer);
    }
    let name = cfg.assistant_name.clone();
    let meeting = Meeting::start(cfg)?;
    let events = meeting.subscribe();
    let mut dash = Dashboard::new(title, chrono::Local::now().format("%-d %b %Y").to_string());
    dash.set_name(name);
    dash.set_folder(meeting.folder().to_path_buf());

    let mut terminal = ratatui::init();
    let drawn = (|| -> Result<()> {
        loop {
            while let Ok(e) = events.try_recv() {
                dash.apply(e);
            }
            terminal.draw(|f| dash.draw(f))?;
            if event::poll(FRAME)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match dash.key(key) {
                    Action::Quit => return Ok(()),
                    Action::Send(cmd) => {
                        if let Err(e) = meeting.send(cmd.clone()) {
                            dash.failed(&cmd, &e);
                        }
                    }
                    Action::Open(target) => {
                        // The system's default app for it.
                        let opened = std::process::Command::new("open")
                            .arg(&target)
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .status();
                        match opened {
                            Ok(s) if s.success() => {}
                            Ok(s) => dash.open_failed(&target, &format!("`open` exited with {s}")),
                            Err(e) => dash.open_failed(&target, &e.to_string()),
                        }
                    }
                    Action::None => {}
                }
            }
            dash.tick();
        }
    })();
    ratatui::restore();
    if notes {
        eprintln!("Finishing the notes…");
    }

    // Stop before reporting a dashboard failure, so the folder is always closed.
    let report = match (drawn, meeting.stop()) {
        (Ok(()), stopped) => stopped?,
        (Err(e), Ok(report)) => {
            return Err(
                e.context(format!("the dashboard failed; meeting saved in {}", report.folder.display()))
            );
        }
        (Err(e), Err(stop)) => {
            return Err(
                e.context(format!("the dashboard failed, and stopping the meeting failed too: {stop:#}"))
            );
        }
    };
    println!(
        "Meeting saved in {} ({} lines, {}).",
        report.folder.display(),
        report.lines,
        mmss(report.duration)
    );
    match (notes, &report.notes_failure) {
        (_, Some(why)) => eprintln!("Notes are incomplete: {why}"),
        (true, None) => println!("Notes: {}", report.folder.join("notes.md").display()),
        (false, None) => {}
    }
    if let Some(why) = &report.context_failure {
        eprintln!("context.md is out of date: {why}");
    }
    Ok(())
}
