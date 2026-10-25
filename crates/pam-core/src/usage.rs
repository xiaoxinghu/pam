//! What a meeting's model calls cost, as pi reports it.
//!
//! pi prices every call from its model catalog, so Pam keeps no prices of its
//! own, and stays right as prices and models change: it adds up the `usage`
//! pi reports for each call, per use and in total, and `meeting.json` keeps
//! the record as `usage`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One model call, as pi reported it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Call {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    /// In US dollars; `None` when pi couldn't price it.
    pub cost: Option<f64>,
}

impl Call {
    /// From pi's `usage` (`input`, `output`, `cacheRead`, `cacheWrite`,
    /// `cost.total`), as an assistant message or a compaction carries it.
    /// `None` when there is none.
    pub fn from_pi(usage: &Value) -> Option<Call> {
        if !usage.is_object() {
            return None;
        }
        let tokens = |key: &str| usage[key].as_u64().unwrap_or(0);
        Some(Call {
            input: tokens("input"),
            output: tokens("output"),
            cache_read: tokens("cacheRead"),
            cache_write: tokens("cacheWrite"),
            cost: usage["cost"]["total"].as_f64(),
        })
    }

    /// The same tokens, with no cost: the model isn't in pi's catalog, so
    /// pi priced it at another model's rates.
    pub fn unpriced(self) -> Call {
        Call { cost: None, ..self }
    }
}

/// What one run used, as far as pi reported it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Used {
    /// pi reported the call.
    Call(Call),
    /// No call was made: pi didn't start.
    Nothing,
    /// pi never reported: it timed out, crashed or exited first, maybe
    /// after the call was made.
    Unknown,
}

/// What a use of the models, or the whole meeting, has spent so far.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Spend {
    /// Model calls: one per scribe run or intent check, and for the agent one
    /// per assistant message and per compaction.
    pub calls: u64,
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    /// In US dollars, at API prices.
    pub cost: f64,
    /// `oauth` when the model's login is a subscription: its cost is what
    /// the calls would cost through the API, not what was billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub login: Option<String>,
}

impl Spend {
    fn add(&mut self, c: &Call) {
        self.calls += 1;
        self.input += c.input;
        self.output += c.output;
        self.cache_read += c.cache_read;
        self.cache_write += c.cache_write;
        self.cost += c.cost.unwrap_or(0.0);
    }

    /// Whether some of its cost is at API prices for a subscription login.
    fn at_api_prices(&self) -> bool {
        self.calls > 0 && self.login.as_deref() == Some("oauth")
    }

    /// Its cost for display: `$0.123`, or `≈$0.123` at API prices for a
    /// subscription login.
    pub fn dollars(&self) -> String {
        let about = if self.at_api_prices() { "≈" } else { "" };
        format!("{about}{}", dollars(self.cost))
    }
}

/// A cost for display: to the cent from a dollar on (`$1.20`), and to a
/// tenth of a cent below it (`$0.004`), since a scribe run costs about that.
pub fn dollars(cost: f64) -> String {
    if cost < 0.9995 { format!("${cost:.3}") } else { format!("${cost:.2}") }
}

/// Who made a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum User {
    Scribe,
    /// The intent check: on the scribe's model unless another is named,
    /// and at no cost on the local one.
    Intent,
    Agent,
}

/// The meeting's tokens and cost so far: `meeting.json`'s `usage`, and the
/// `Usage` event front-ends get after each call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub scribe: Spend,
    pub intent: Spend,
    pub agent: Spend,
    /// The sum of the three.
    pub total: Spend,
    /// Some calls are missing or have no cost: pi couldn't price them, or a
    /// run ended before pi reported what it used.
    pub partial: bool,
}

impl Usage {
    /// Nothing spent yet, with the scribe's, the intent check's and the
    /// agent's logins as `pi auth check` named them (`oauth`, `api_key`).
    pub fn new(scribe_login: Option<&str>, intent_login: Option<&str>, agent_login: Option<&str>) -> Usage {
        let spend = |login: Option<&str>| Spend { login: login.map(str::to_string), ..Spend::default() };
        Usage {
            scribe: spend(scribe_login),
            intent: spend(intent_login),
            agent: spend(agent_login),
            ..Usage::default()
        }
    }

    /// Counts a call `who` made. One pi couldn't price makes the total partial.
    pub fn add(&mut self, who: User, call: &Call) {
        let spend = match who {
            User::Scribe => &mut self.scribe,
            User::Intent => &mut self.intent,
            User::Agent => &mut self.agent,
        };
        spend.add(call);
        self.total.add(call);
        if call.cost.is_none() {
            self.partial = true;
        }
    }

    /// A run ended before pi reported all it used: it was stopped, timed
    /// out or crashed mid-call.
    pub fn lost(&mut self) {
        self.partial = true;
    }

    /// Whether some of the cost is at API prices for a subscription login.
    fn at_api_prices(&self) -> bool {
        [&self.scribe, &self.intent, &self.agent].iter().any(|s| s.at_api_prices())
    }

    /// The total for display: `$0.200`, with `≈` in front when some of it
    /// is at API prices for a subscription login, and `+` after it when partial.
    pub fn label(&self) -> String {
        let about = if self.at_api_prices() { "≈" } else { "" };
        let more = if self.partial { "+" } else { "" };
        format!("{about}{}{more}", dollars(self.total.cost))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(input: u64, output: u64, cost: Option<f64>) -> Call {
        Call { input, output, cache_read: 10, cache_write: 1, cost }
    }

    #[test]
    fn reads_pi_usage() {
        let usage = json!({"input": 97, "output": 4, "cacheRead": 2, "cacheWrite": 3, "totalTokens": 106,
                           "cost": {"input": 0.0000097, "output": 0.000002, "total": 0.0000117}});
        assert_eq!(
            Call::from_pi(&usage),
            Some(Call { input: 97, output: 4, cache_read: 2, cache_write: 3, cost: Some(0.0000117) })
        );
        assert_eq!(Call::from_pi(&json!({"input": 5})).unwrap().cost, None);
        assert_eq!(Call::from_pi(&Value::Null), None);
    }

    #[test]
    fn adds_calls_per_use_and_in_total() {
        let mut u = Usage::new(Some("api_key"), Some("api_key"), Some("oauth"));
        assert_eq!(u.label(), "$0.000", "nothing at API prices before a call");
        u.add(User::Scribe, &call(100, 20, Some(0.12)));
        u.add(User::Intent, &call(5, 1, Some(0.001)));
        u.add(User::Agent, &call(40, 30, Some(0.08)));
        u.add(User::Agent, &call(1, 2, Some(0.0)));
        assert_eq!((u.scribe.calls, u.intent.calls, u.agent.calls, u.total.calls), (1, 1, 2, 4));
        assert_eq!(
            (u.total.input, u.total.output, u.total.cache_read, u.total.cache_write),
            (146, 53, 40, 4)
        );
        assert!((u.total.cost - 0.201).abs() < 1e-9);
        assert!(!u.partial);
        assert_eq!(u.label(), "≈$0.201");
        assert_eq!((u.scribe.dollars(), u.agent.dollars()), ("$0.120".into(), "≈$0.080".into()));

        // A call pi couldn't price counts its tokens, not a cost.
        u.add(User::Scribe, &call(7, 0, None));
        assert_eq!((u.scribe.calls, u.scribe.input), (2, 107));
        assert!(u.partial);
        assert_eq!(u.label(), "≈$0.201+");
        // From a dollar on, to the cent.
        u.add(User::Agent, &call(1, 1, Some(1.0)));
        assert_eq!(u.label(), "≈$1.20+");
        assert_eq!(
            (dollars(0.9994), dollars(0.9996), dollars(12.345)),
            ("$0.999".into(), "$1.00".into(), "$12.35".into())
        );
    }

    #[test]
    fn the_record_reads_as_the_spec_writes_it() {
        let u = Usage::new(None, None, Some("oauth"));
        let v = serde_json::to_value(&u).unwrap();
        assert_eq!(
            v["agent"],
            json!({"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
                                      "cost": 0.0, "login": "oauth"})
        );
        assert_eq!(v["partial"], false);
        assert_eq!(serde_json::from_value::<Usage>(v).unwrap(), u);
    }
}
