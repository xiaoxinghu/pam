//! Languages from scripts, shared by the adapters: what an adapter
//! does when its engine gives an utterance no language, or one its text
//! contradicts. The NeMo adapter and the clip wrapper both use [`fill`].

use crate::{Word, is_cjk_language};

/// The language a text's script gives away, for an utterance the model gave
/// no language tag or a tag its text contradicts. Nemotron 3.5 tags an
/// utterance's language by emitting a `<zh-CN>`-style token, and often
/// doesn't. Any kana means Japanese; more Hangul than Han
/// characters, Korean; otherwise Han characters mean Chinese. Latin and the
/// other scripts are shared by too many languages to tell.
fn script_language(text: &str) -> Option<&'static str> {
    let count = |is: fn(char) -> bool| text.chars().filter(|c| is(*c)).count();
    let (kana, hangul, han) = (count(crate::is_kana), count(crate::is_hangul), count(crate::is_han));
    match () {
        _ if kana > 0 => Some("ja-JP"),
        _ if hangul > han => Some("ko-KR"),
        _ if han > 0 => Some("zh-CN"),
        _ => None,
    }
}

/// The language of an utterance with text `text` that was tagged `tag`: the
/// tag, unless it is Chinese, Japanese or Korean and the text's script says
/// another of the three, or there is no tag. The model's tag can be wrong
/// where its first syllables were: a Mandarin turn opening with 下周末
/// ("next weekend") came out as "샤르朋友结婚, …", tagged `ko-KR`.
fn utterance_language(text: &str, tag: Option<&str>) -> Option<String> {
    let script = script_language(text);
    let lang = match tag {
        Some(t) if is_cjk_language(t) => {
            // Han characters alone are written in Japanese and Cantonese too.
            let han = |t: &str| ["ja", "yue"].contains(&t.split('-').next().unwrap_or(t));
            let fits = |s: &str| s[..2] == t[..2] || (s == "zh-CN" && han(t));
            script.filter(|s| !fits(s)).unwrap_or(t)
        }
        Some(t) => t,
        None => script?,
    };
    Some(lang.to_string())
}

/// The language of one word of an utterance in `lang`. An utterance can hold
/// two turns when the second speaker starts without a pause (an English turn
/// after a Mandarin one has come out right, tagged `zh-CN`). So a
/// CJK word in an utterance in another language takes its script's language,
/// and a word in letters in a Chinese, Japanese or Korean utterance gets
/// none: it may be an English turn, or a name inside a Mandarin sentence.
/// Anything else keeps the utterance's language.
fn word_language(word: &str, lang: Option<&str>) -> Option<String> {
    let cjk = lang.is_some_and(is_cjk_language);
    if word.chars().any(crate::is_cjk) {
        return if cjk { lang.map(String::from) } else { script_language(word).map(String::from) };
    }
    if cjk && word.chars().any(char::is_alphabetic) {
        return None;
    }
    lang.map(String::from)
}

/// Gives the words of one utterance that have no language one: `tag`, the
/// language the engine or the configuration gives the utterance, checked
/// against the script of their text (see [`utterance_language`]), then word
/// by word (see [`word_language`]). Words that have a language keep it.
pub fn fill(words: &mut [Word], tag: Option<&str>) {
    let text: String = words.iter().filter(|w| w.lang.is_none()).map(|w| w.text.as_str()).collect();
    let lang = utterance_language(&text, tag);
    for w in words.iter_mut().filter(|w| w.lang.is_none()) {
        w.lang = word_language(&w.text, lang.as_deref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts_tell_only_cjk_languages() {
        assert_eq!(script_language("没问题, 我周三之前会把测试报告发给大家"), Some("zh-CN"));
        assert_eq!(script_language("OK, 33 号"), Some("zh-CN"));
        assert_eq!(script_language("会議は三時からです"), Some("ja-JP"));
        assert_eq!(script_language("안녕하세요"), Some("ko-KR"));
        assert_eq!(script_language("Hola, ¿qué tal?"), None);
        assert_eq!(script_language("Привет"), None);
        assert_eq!(script_language(""), None);
        assert_eq!(script_language("𠮷野家〇"), Some("zh-CN"), "Han outside the BMP, and 〇");
        // Two Hangul syllables misheard at the start of a Mandarin turn.
        assert_eq!(script_language("샤르朋友结婚, 我们要开车去另一个城市参加婚礼"), Some("zh-CN"));
    }

    #[test]
    fn a_cjk_tag_the_text_contradicts_gives_way_to_the_script() {
        let text = "샤르朋友结婚, 我们要开车去另一个城市参加婚礼";
        assert_eq!(utterance_language(text, Some("ko-KR")).as_deref(), Some("zh-CN"));
        assert_eq!(
            utterance_language("会議室", Some("ja-JP")).as_deref(),
            Some("ja-JP"),
            "kanji alone may be Japanese"
        );
        assert_eq!(utterance_language("会議は三時から", Some("zh-CN")).as_deref(), Some("ja-JP"));
        assert_eq!(
            utterance_language("我哋聽日去飲茶", Some("yue")).as_deref(),
            Some("yue"),
            "Cantonese is written in Han"
        );
        assert_eq!(utterance_language("다음 주에", Some("yue")).as_deref(), Some("ko-KR"));
        assert_eq!(utterance_language("OK, then let's keep Friday", Some("zh-CN")).as_deref(), Some("zh-CN"));
        assert_eq!(
            utterance_language("没问题", Some("en-US")).as_deref(),
            Some("en-US"),
            "words are checked one by one"
        );
        assert_eq!(utterance_language("没问题", None).as_deref(), Some("zh-CN"));
        assert_eq!(utterance_language("Great", None), None);
    }

    #[test]
    fn a_word_whose_script_the_tag_cant_write_gets_its_own_language() {
        let zh = Some("zh-CN");
        assert_eq!(word_language("大家好,", zh).as_deref(), Some("zh-CN"));
        assert_eq!(word_language("Friday", zh), None);
        assert_eq!(word_language("33", zh).as_deref(), Some("zh-CN"));
        assert_eq!(word_language("没问题,", Some("en-US")).as_deref(), Some("zh-CN"));
        assert_eq!(word_language("会議", Some("ja-JP")).as_deref(), Some("ja-JP"));
        assert_eq!(word_language("飲茶", Some("yue")).as_deref(), Some("yue"));
        assert_eq!(word_language("dim sum", Some("yue")), None);
        assert_eq!(word_language("Friday", Some("en-US")).as_deref(), Some("en-US"));
        assert_eq!(word_language("没问题", None).as_deref(), Some("zh-CN"));
        assert_eq!(word_language("Friday", None), None);
    }

    #[test]
    fn fill_labels_only_words_without_a_language() {
        let w = |text: &str, lang: Option<&str>| Word {
            text: text.into(),
            lang: lang.map(String::from),
            ..Word::default()
        };
        let mut words = [w("Pam", None), w("没问题", None), w("OK", Some("en-GB"))];
        fill(&mut words, Some("en-US"));
        let langs: Vec<Option<&str>> = words.iter().map(|w| w.lang.as_deref()).collect();
        assert_eq!(langs, [Some("en-US"), Some("zh-CN"), Some("en-GB")]);
        let mut words = [w("大家好", None), w("Friday", None)];
        fill(&mut words, None);
        assert_eq!(words.iter().map(|w| w.lang.as_deref()).collect::<Vec<_>>(), [Some("zh-CN"), None]);
    }
}
