// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Single-byte Mac OS script encodings (Mac OS Roman and its siblings),
//! from the mapping tables Apple published through Unicode.org.

mod tables;

use tables::{
    CENTRAL_EUROPEAN_DECODE, CENTRAL_EUROPEAN_ENCODE, CROATIAN_DECODE, CROATIAN_ENCODE,
    CYRILLIC_DECODE, CYRILLIC_ENCODE, GREEK_DECODE, GREEK_ENCODE, ICELANDIC_DECODE,
    ICELANDIC_ENCODE, ROMAN_DECODE, ROMAN_ENCODE, ROMANIAN_DECODE, ROMANIAN_ENCODE, TURKISH_DECODE,
    TURKISH_ENCODE,
};

/// A single-byte Mac OS text encoding, used to decode and encode classic HFS
/// names (and, later, text previews).
///
/// Bytes 0x00-0x7F are ASCII in every variant. A byte that the mapping file
/// leaves undefined decodes to U+FFFD (Mac OS Turkish 0xF5 is the only one),
/// and U+FFFD never encodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub enum Encoding {
    /// Mac OS Roman, the default.
    #[default]
    Roman,
    CentralEuropean,
    Cyrillic,
    Greek,
    Turkish,
    Croatian,
    Icelandic,
    Romanian,
}

impl Encoding {
    /// Every encoding, in declaration order.
    pub const ALL: [Encoding; 8] = [
        Encoding::Roman,
        Encoding::CentralEuropean,
        Encoding::Cyrillic,
        Encoding::Greek,
        Encoding::Turkish,
        Encoding::Croatian,
        Encoding::Icelandic,
        Encoding::Romanian,
    ];

    /// The decode table for bytes 0x80-0xFF and the encode table (sorted by
    /// character) for the characters it maps.
    fn tables(self) -> (&'static [char; 128], &'static [(char, u8)]) {
        match self {
            Encoding::Roman => (&ROMAN_DECODE, ROMAN_ENCODE),
            Encoding::CentralEuropean => (&CENTRAL_EUROPEAN_DECODE, CENTRAL_EUROPEAN_ENCODE),
            Encoding::Cyrillic => (&CYRILLIC_DECODE, CYRILLIC_ENCODE),
            Encoding::Greek => (&GREEK_DECODE, GREEK_ENCODE),
            Encoding::Turkish => (&TURKISH_DECODE, TURKISH_ENCODE),
            Encoding::Croatian => (&CROATIAN_DECODE, CROATIAN_ENCODE),
            Encoding::Icelandic => (&ICELANDIC_DECODE, ICELANDIC_ENCODE),
            Encoding::Romanian => (&ROMANIAN_DECODE, ROMANIAN_ENCODE),
        }
    }

    /// Appends the decoded text. Every byte decodes; nothing fails. Bytes
    /// 0x00 and `:` decode as themselves (the caller sanitises names).
    pub fn decode_into(self, bytes: &[u8], out: &mut String) {
        let (high, _) = self.tables();
        out.reserve(bytes.len());
        for &byte in bytes {
            out.push(if byte < 0x80 {
                char::from(byte)
            } else {
                // The table has 128 entries, so `get` always succeeds.
                high.get(usize::from(byte & 0x7F))
                    .copied()
                    .unwrap_or('\u{FFFD}')
            });
        }
    }

    /// Encodes `text`, or returns `None` when a character has no byte in this
    /// encoding. This is the exact inverse of `decode_into`: a byte string
    /// that decodes without U+FFFD encodes back to itself.
    pub fn encode(self, text: &str) -> Option<Vec<u8>> {
        let (_, encode) = self.tables();
        let mut out = Vec::with_capacity(text.len());
        for c in text.chars() {
            if c.is_ascii() {
                out.push(c as u8);
            } else {
                let i = encode.binary_search_by_key(&c, |&(ch, _)| ch).ok()?;
                out.push(encode.get(i)?.1);
            }
        }
        Some(out)
    }

    /// The display name, for example "Mac OS Roman".
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Roman => "Mac OS Roman",
            Encoding::CentralEuropean => "Mac OS Central European",
            Encoding::Cyrillic => "Mac OS Cyrillic",
            Encoding::Greek => "Mac OS Greek",
            Encoding::Turkish => "Mac OS Turkish",
            Encoding::Croatian => "Mac OS Croatian",
            Encoding::Icelandic => "Mac OS Icelandic",
            Encoding::Romanian => "Mac OS Romanian",
        }
    }

    /// The encoding for an Apple script code: smRoman 0, smGreek 6,
    /// smCyrillic 7 and smCentralEuroRoman 29. Turkish, Croatian, Icelandic
    /// and Romanian are Roman-script region variants with no script code of
    /// their own, so they are only chosen explicitly. Every other code,
    /// including the multi-byte scripts (not supported yet), gives `None`.
    pub fn from_script_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Encoding::Roman),
            6 => Some(Encoding::Greek),
            7 => Some(Encoding::Cyrillic),
            29 => Some(Encoding::CentralEuropean),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(enc: Encoding, bytes: &[u8]) -> String {
        let mut s = String::new();
        enc.decode_into(bytes, &mut s);
        s
    }

    fn one(enc: Encoding, byte: u8) -> char {
        let s = decode(enc, &[byte]);
        let mut chars = s.chars();
        let c = chars.next().expect("one char");
        assert!(chars.next().is_none());
        c
    }

    #[test]
    fn round_trips_all_256_bytes() {
        for enc in Encoding::ALL {
            for byte in 0..=255u8 {
                let c = one(enc, byte);
                if c == '\u{FFFD}' {
                    assert_eq!(enc.encode("\u{FFFD}"), None, "{enc:?}");
                    continue;
                }
                assert_eq!(
                    enc.encode(&c.to_string()),
                    Some(vec![byte]),
                    "{enc:?} byte {byte:#04x}"
                );
            }
        }
    }

    #[test]
    fn whole_range_round_trips_as_one_string() {
        let all: Vec<u8> = (0..=255).collect();
        for enc in Encoding::ALL {
            let text = decode(enc, &all);
            if text.contains('\u{FFFD}') {
                assert_eq!(enc.encode(&text), None);
            } else {
                assert_eq!(enc.encode(&text), Some(all.clone()), "{enc:?}");
            }
        }
    }

    #[test]
    fn tables_are_strictly_sorted_and_complete() {
        for enc in Encoding::ALL {
            let (high, encode) = enc.tables();
            assert!(
                encode.windows(2).all(|w| w[0].0 < w[1].0),
                "{enc:?} encode table not strictly sorted"
            );
            let defined = high.iter().filter(|&&c| c != '\u{FFFD}').count();
            assert_eq!(encode.len(), defined, "{enc:?}");
            assert!(encode.iter().all(|&(c, b)| b >= 0x80 && c != '\u{FFFD}'));
        }
    }

    #[test]
    fn ascii_is_identity_in_every_encoding() {
        let all: Vec<u8> = (0..0x80).collect();
        for enc in Encoding::ALL {
            let text = decode(enc, &all);
            assert!(text.is_ascii());
            assert_eq!(text.as_bytes(), all.as_slice());
            assert_eq!(enc.encode(&text), Some(all.clone()));
        }
    }

    #[test]
    fn roman_spot_values() {
        assert_eq!(one(Encoding::Roman, 0x8E), '\u{00E9}');
        assert_eq!(one(Encoding::Roman, 0xA5), '\u{2022}');
        assert_eq!(one(Encoding::Roman, 0xDB), '\u{20AC}');
        assert_eq!(one(Encoding::Roman, 0xF0), '\u{F8FF}');
        assert_eq!(Encoding::Roman.encode("\u{F8FF}"), Some(vec![0xF0]));
        assert_eq!(
            Encoding::Roman.encode("caf\u{00E9}"),
            Some(b"caf\x8E".to_vec())
        );
    }

    #[test]
    fn spot_values_from_the_other_mapping_files() {
        assert_eq!(one(Encoding::Cyrillic, 0x80), '\u{0410}');
        assert_eq!(one(Encoding::Cyrillic, 0xFF), '\u{20AC}');
        assert_eq!(one(Encoding::CentralEuropean, 0x81), '\u{0100}');
        assert_eq!(one(Encoding::Greek, 0xE1), '\u{03B1}');
        assert_eq!(one(Encoding::Greek, 0xFF), '\u{00AD}');
        assert_eq!(one(Encoding::Turkish, 0xDA), '\u{011E}');
        assert_eq!(one(Encoding::Croatian, 0xC6), '\u{0106}');
        assert_eq!(one(Encoding::Icelandic, 0xDC), '\u{00D0}');
        assert_eq!(one(Encoding::Romanian, 0xAF), '\u{0218}');
    }

    #[test]
    fn turkish_undefined_byte_is_the_replacement_character() {
        assert_eq!(one(Encoding::Turkish, 0xF5), '\u{FFFD}');
        // Only Turkish 0xF5 is undefined in these files.
        for enc in Encoding::ALL {
            let undefined = (0..=255u8).filter(|&b| one(enc, b) == '\u{FFFD}').count();
            assert_eq!(undefined, usize::from(enc == Encoding::Turkish), "{enc:?}");
        }
    }

    #[test]
    fn unmappable_and_empty() {
        for enc in Encoding::ALL {
            assert_eq!(enc.encode("\u{4E2D}"), None);
            assert_eq!(enc.encode("a\u{4E2D}b"), None);
            assert_eq!(enc.encode("\u{FFFD}"), None);
            assert_eq!(enc.encode(""), Some(vec![]));
            assert_eq!(decode(enc, b""), "");
        }
    }

    #[test]
    fn decode_appends_and_keeps_colon_and_nul() {
        let mut s = String::from("x");
        Encoding::Roman.decode_into(b"a:b\0\x8E", &mut s);
        assert_eq!(s, "xa:b\0\u{00E9}");
    }

    #[test]
    fn script_codes() {
        assert_eq!(Encoding::from_script_code(0), Some(Encoding::Roman));
        assert_eq!(Encoding::from_script_code(6), Some(Encoding::Greek));
        assert_eq!(Encoding::from_script_code(7), Some(Encoding::Cyrillic));
        assert_eq!(
            Encoding::from_script_code(29),
            Some(Encoding::CentralEuropean)
        );
        for code in 0..=255u8 {
            if ![0, 6, 7, 29].contains(&code) {
                assert_eq!(Encoding::from_script_code(code), None, "{code}");
            }
        }
    }

    #[test]
    fn names_are_distinct_and_all_is_complete() {
        let names: std::collections::BTreeSet<_> = Encoding::ALL.iter().map(|e| e.name()).collect();
        assert_eq!(names.len(), 8);
        assert_eq!(Encoding::Roman.name(), "Mac OS Roman");
        assert_eq!(Encoding::CentralEuropean.name(), "Mac OS Central European");
        assert_eq!(Encoding::default(), Encoding::Roman);
        // Exhaustive match: adding a variant breaks this until ALL is updated.
        for (position, e) in Encoding::ALL.into_iter().enumerate() {
            let expected = match e {
                Encoding::Roman => 0,
                Encoding::CentralEuropean => 1,
                Encoding::Cyrillic => 2,
                Encoding::Greek => 3,
                Encoding::Turkish => 4,
                Encoding::Croatian => 5,
                Encoding::Icelandic => 6,
                Encoding::Romanian => 7,
            };
            assert_eq!(position, expected);
        }
    }
}
