//! The fast path of naming from introductions: a pattern match on a
//! final line.
//!
//! It matches "I'm X", "I am X", "my name is X", "this is X" and "X here".
//! A name is one or two capitalised words, so it relies on the transcriber's
//! capitalisation. It errs towards missing an intro: the scribe confirms and
//! catches harder phrasings later, and every rename can be undone.

/// Capitalised words that start sentences or follow "I'm" without being names.
const NOT_NAMES: &[&str] = &[
    "Okay",
    "Ok",
    "Right",
    "Yes",
    "Yeah",
    "No",
    "Nope",
    "So",
    "Well",
    "Sorry",
    "Thanks",
    "Hi",
    "Hello",
    "Hey",
    "Just",
    "Not",
    "Also",
    "Still",
    "Here",
    "There",
    "This",
    "That",
    "These",
    "Those",
    "We",
    "You",
    "They",
    "He",
    "She",
    "It",
    "Me",
    "Us",
    "Them",
    "My",
    "Our",
    "Your",
    "Everyone",
    "Everybody",
    "Anyone",
    "Someone",
    "Somebody",
    "Nobody",
    "Look",
    "Click",
    "Sign",
    "Start",
    "Stop",
    "Over",
    "Down",
    "Up",
    "Out",
    "In",
    "Back",
    "Come",
    "Get",
    "Put",
    "Sit",
    "Stay",
    "Wait",
    "Good",
    "Great",
    "Fine",
    "Sure",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// The name a line introduces its speaker by, if it matches a pattern.
/// `assistant` is the assistant's name, which is never a person's.
pub fn introduced_name(text: &str, assistant: &str) -> Option<String> {
    let text = text.replace('’', "'");
    let words: Vec<&str> = text.split_whitespace().collect();
    let lower: Vec<String> = words.iter().map(|w| bare(w).to_lowercase()).collect();
    let is = |i: usize, w: &str| lower.get(i).is_some_and(|l| l == w);
    // Whether words i..i+n spell `trigger` with no punctuation inside or after it.
    let says = |i: usize, trigger: &[&str]| {
        trigger.iter().enumerate().all(|(k, w)| is(i + k, w) && !ends_clause(words[i + k]))
    };
    for i in 0..words.len() {
        // "I'm X", "I am X", "my name is X": any capitalised name that follows.
        let after = [&["i'm"][..], &["i", "am"], &["my", "name", "is"]]
            .iter()
            .find(|t| says(i, t))
            .map(|t| i + t.len());
        if let Some(name) = after.and_then(|at| name_at(&words, at, assistant)) {
            return Some(name.text);
        }
        // "this is X": only when the name ends its clause ("This is Jordan, design."),
        // so "this is Figma's new flow" is not an intro.
        if says(i, &["this", "is"])
            && let Some(name) = name_at(&words, i + 2, assistant)
            && (name.ends_clause || is(name.next, "from") || is(name.next, "speaking"))
        {
            return Some(name.text);
        }
        // "X here": a one-word name starting a sentence or after a greeting,
        // with "here" ending the clause ("Marcus here, engineering.").
        if is(i + 1, "here")
            && (i == 0 || ends_sentence(words[i - 1]) || words[i - 1].ends_with(','))
            && !ends_clause(words[i])
            && (i + 2 == words.len() || ends_clause(words[i + 1]))
            && let Some(name) = name_at(&words, i, assistant)
            && name.next == i + 1
        {
            return Some(name.text);
        }
    }
    None
}

struct Found {
    text: String,
    /// The index of the first word after the name.
    next: usize,
    /// Whether punctuation follows the name.
    ends_clause: bool,
}

/// One or two capitalised words starting at `at`, such as "Alice" or "Alice Chen".
/// None when the assistant's name (one word or more) starts the candidate or
/// follows its first word: "I'm Pam", "I'm Project Pam" are not intros.
fn name_at(words: &[&str], at: usize, assistant: &str) -> Option<Found> {
    let first = *words.get(at)?;
    if spells(words, at, assistant) || (!ends_clause(first) && spells(words, at + 1, assistant)) {
        return None;
    }
    if !looks_like_name(first) {
        return None;
    }
    let mut parts = vec![bare(first)];
    let mut next = at + 1;
    if !ends_clause(first)
        && let Some(second) = words.get(next)
        && looks_like_name(second)
    {
        parts.push(bare(second));
        next += 1;
    }
    Some(Found { text: parts.join(" "), next, ends_clause: ends_clause(words[next - 1]) })
}

/// Whether the words from `at` spell `name`, ignoring case and punctuation around them.
fn spells(words: &[&str], at: usize, name: &str) -> bool {
    let want: Vec<String> = name.split_whitespace().map(|w| bare(w).to_lowercase()).collect();
    !want.is_empty()
        && want
            .iter()
            .enumerate()
            .all(|(k, w)| words.get(at + k).is_some_and(|x| bare(x).to_lowercase() == *w))
}

/// A capitalised word of letters, with at least one lowercase letter (so not
/// "I" or "OK"), that isn't a possessive, a contraction ("I'm", "Let's") or a
/// common non-name.
fn looks_like_name(word: &str) -> bool {
    let w = bare(word);
    let mut chars = w.chars();
    let Some(first) = chars.next() else { return false };
    first.is_uppercase()
        && w.chars().any(char::is_lowercase)
        && w.chars().all(|c| c.is_alphabetic() || c == '\'' || c == '-')
        && !["'s", "'m", "'re", "'ll", "'ve", "'d", "n't"].iter().any(|end| w.ends_with(end))
        && !NOT_NAMES.contains(&w)
}

/// The word without punctuation around it.
fn bare(word: &str) -> &str {
    word.trim_matches(|c: char| !c.is_alphanumeric())
}

fn ends_clause(word: &str) -> bool {
    word.ends_with([',', '.', '!', '?', ';', ':'])
}

fn ends_sentence(word: &str) -> bool {
    word.ends_with(['.', '!', '?'])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(text: &str) -> Option<String> {
        introduced_name(text, "Pam")
    }

    #[test]
    fn matches_introductions() {
        let cases = [
            ("Quick intros for the bot. I'm Alice, I run product.", "Alice"),
            ("And I'm Jordan, design.", "Jordan"),
            ("Thanks. I’m Sam, by the way, from sales.", "Sam"),
            ("Hi everyone, I am Priya and I lead data.", "Priya"),
            ("My name is Alice Chen.", "Alice Chen"),
            ("my name is Mary-Jane O'Brien", "Mary-Jane O'Brien"),
            ("This is Jordan, design.", "Jordan"),
            ("Hi, this is Sam from sales.", "Sam"),
            ("Marcus here, engineering.", "Marcus"),
            ("Hi all. Marcus here.", "Marcus"),
            ("Morning, Marcus here", "Marcus"),
            ("I'm Alice I run product", "Alice"),
        ];
        for (text, want) in cases {
            assert_eq!(name(text).as_deref(), Some(want), "{text}");
        }
    }

    #[test]
    fn ignores_what_only_looks_like_one() {
        let cases = [
            "Sorry I'm late.",
            "I'm not sure that's right.",
            "I'm OK with that.",
            "I'm Sorry, what?",
            "Okay, I'm here.",
            "Let's here.",
            "This is great.",
            "This is Figma's new flow.",
            "This is Google Docs which we use.",
            "This is Monday's agenda.",
            "Click here.",
            "Okay here.",
            "Put it here.",
            "We're deciding between the three-step flow and a single page.",
            "Marcus, engineering.",
            "I'm Pam.",
            "This is Pam.",
            "I am. Sam will join later.",
            "",
        ];
        for text in cases {
            assert_eq!(name(text), None, "{text}");
        }
    }

    #[test]
    fn the_assistant_name_is_configurable() {
        assert_eq!(introduced_name("I'm Ada.", "Ada"), None);
        assert_eq!(introduced_name("I'm Pam.", "Ada").as_deref(), Some("Pam"));
        for text in ["I'm Project Pam.", "This is Project  Pam, hello.", "Project Pam here."] {
            assert_eq!(introduced_name(text, "Project  Pam"), None, "{text}");
        }
        assert_eq!(introduced_name("I'm Project Lead.", "Project Pam").as_deref(), Some("Project Lead"));
        // A one-word assistant name after a capitalised word rejects the candidate, not just the word.
        for text in ["I'm Project Pam.", "This is Project Pam, hello.", "My name is Ask Pam."] {
            assert_eq!(introduced_name(text, "Pam"), None, "{text}");
        }
        assert_eq!(introduced_name("I'm Alice, Pam can you note that?", "Pam").as_deref(), Some("Alice"));
    }
}
