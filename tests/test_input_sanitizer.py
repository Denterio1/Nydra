"""
Tests for src/input_sanitizer.py - the security layer's input sanitizer.

No network access: DNS is always faked through the `resolver=` argument.
"""

import socket

import pytest

from src.input_sanitizer import (
    ENV_ALLOW_PRIVATE_URLS,
    InputRejectedError,
    SanitizeResult,
    neutralize_csv_cell,
    neutralize_dataframe,
    sanitize_filename,
    sanitize_text,
    validate_url,
)

CSV_ONLY = {"csv"}


def public(host):
    return ["8.8.8.8"]


def must_not_resolve(host):
    raise AssertionError(f"resolver must not be called (got {host!r})")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_ALLOW_PRIVATE_URLS, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
class TestResultType:
    def test_unwrap_returns_value_when_ok(self):
        assert sanitize_text("hello").unwrap() == "hello"

    def test_unwrap_raises_on_rejection(self):
        with pytest.raises(InputRejectedError) as exc:
            sanitize_filename("../../etc/passwd", CSV_ONLY).unwrap()
        assert isinstance(exc.value.result, SanitizeResult)

    def test_rejected_result_has_no_value_and_a_reason(self):
        r = sanitize_filename(None)
        assert r.rejected and not r.ok and r.value is None and r.reasons


# ─────────────────────────────────────────────────────────────────────────────
class TestFilename:
    def test_clean_name_unchanged(self):
        r = sanitize_filename("data.csv", CSV_ONLY)
        assert r.ok and r.value == "data.csv" and not r.changed

    def test_unix_traversal_reduced_to_basename(self):
        r = sanitize_filename("../../etc/evil.csv", CSV_ONLY)
        assert r.ok and r.value == "evil.csv" and r.changed

    def test_windows_traversal_reduced_to_basename(self):
        r = sanitize_filename("..\\..\\evil.csv", CSV_ONLY)
        assert r.ok and r.value == "evil.csv"

    def test_fullwidth_slash_traversal(self):
        r = sanitize_filename("..／..／x.csv", CSV_ONLY)
        assert r.ok and r.value == "x.csv"

    def test_absolute_path_without_extension_rejected(self):
        assert sanitize_filename("/etc/passwd", CSV_ONLY).rejected

    def test_trailing_separator_rejected(self):
        assert sanitize_filename("folder/", CSV_ONLY).rejected

    def test_null_byte_rejected(self):
        assert sanitize_filename("a.csv\x00.exe", CSV_ONLY).rejected

    @pytest.mark.parametrize("name", ["CON.csv", "nul.csv", "com1.csv", "LPT9.csv", "aux.csv"])
    def test_windows_reserved_names_rejected(self, name):
        assert sanitize_filename(name, CSV_ONLY).rejected

    @pytest.mark.parametrize("name", ["shell.php", "archive.csv.exe", "noext", "data.csv.php"])
    def test_disallowed_extension_rejected(self, name):
        assert sanitize_filename(name, CSV_ONLY).rejected

    def test_extension_check_is_case_insensitive_and_keeps_case(self):
        r = sanitize_filename("DATA.CSV", CSV_ONLY)
        assert r.ok and r.value == "DATA.CSV"

    def test_trailing_dots_and_spaces_removed(self):
        r = sanitize_filename("data.csv. ", CSV_ONLY)
        assert r.ok and r.value == "data.csv"

    def test_hidden_file_dot_prefix_removed(self):
        r = sanitize_filename(".env.csv", CSV_ONLY)
        assert r.ok and r.value == "env.csv"

    @pytest.mark.parametrize("name", ["..", "...", ".csv", " ", ""])
    def test_degenerate_names_rejected(self, name):
        assert sanitize_filename(name, CSV_ONLY).rejected

    def test_control_characters_removed(self):
        r = sanitize_filename("da\x07ta.csv", CSV_ONLY)
        assert r.ok and r.value == "data.csv"

    def test_unsafe_characters_replaced(self):
        r = sanitize_filename('a<b>:c|d?.csv', CSV_ONLY)
        assert r.ok and not any(c in r.value for c in '<>:"|?*')

    def test_long_name_truncated_keeping_extension(self):
        r = sanitize_filename("a" * 500 + ".csv", CSV_ONLY, max_length=50)
        assert r.ok and len(r.value) == 50 and r.value.endswith(".csv")

    def test_no_extension_policy_allows_plain_names(self):
        assert sanitize_filename("notes").ok

    def test_arabic_filename_preserved(self):
        r = sanitize_filename("بيانات.csv", CSV_ONLY)
        assert r.ok and r.value == "بيانات.csv" and not r.changed

    def test_extensions_given_with_leading_dot(self):
        assert sanitize_filename("a.csv", {".csv"}).ok

    @pytest.mark.parametrize("bad", [None, 123, b"a.csv", ["a.csv"]])
    def test_non_string_rejected(self, bad):
        assert sanitize_filename(bad).rejected


# ─────────────────────────────────────────────────────────────────────────────
class TestText:
    def test_clean_text_unchanged(self):
        r = sanitize_text("What is the mean of column price?")
        assert r.ok and not r.changed and r.reasons == []

    def test_null_bytes_removed(self):
        r = sanitize_text("hel\x00lo")
        assert r.value == "hello" and r.changed

    def test_newlines_and_tabs_kept_other_controls_removed(self):
        r = sanitize_text("a\nb\tc\x07d\x1b")
        assert r.value == "a\nb\tcd"

    def test_crlf_normalized(self):
        assert sanitize_text("a\r\nb").value == "a\nb"

    def test_lone_carriage_return_removed(self):
        assert sanitize_text("a\rb").value == "ab"

    def test_bidi_override_removed(self):
        assert sanitize_text("admin\u202eevil").value == "adminevil"

    @pytest.mark.parametrize("ch", ["\u202a", "\u202b", "\u202c", "\u202d", "\u2066", "\u2067", "\u2068", "\u2069"])
    def test_all_bidi_controls_removed(self, ch):
        assert sanitize_text(f"a{ch}b").value == "ab"

    def test_zero_width_space_and_bom_removed(self):
        assert sanitize_text("a\u200bb\ufeffc\u2060d").value == "abcd"

    def test_arabic_and_persian_shaping_chars_preserved(self):
        for s in ("مرحبا\u200f بالعالم", "می\u200cخواهم", "hello\u200e world"):
            r = sanitize_text(s)
            assert r.ok and r.value == s and not r.changed

    def test_too_long_rejected_by_default(self):
        assert sanitize_text("x" * 11, max_length=10).rejected

    def test_too_long_can_truncate(self):
        r = sanitize_text("x" * 50, max_length=10, on_too_long="truncate")
        assert r.ok and len(r.value) == 10 and r.changed

    def test_empty_and_whitespace_rejected_unless_allowed(self):
        assert sanitize_text("").rejected
        assert sanitize_text("  \n ").rejected
        assert sanitize_text("", allow_empty=True).ok

    def test_newlines_disallowed_become_spaces(self):
        assert sanitize_text("a\nb\t\tc", allow_newlines=False).value == "a b c"

    def test_lone_surrogate_removed(self):
        r = sanitize_text("a\ud800b")
        assert r.value == "ab"

    def test_nfc_normalization(self):
        r = sanitize_text("e\u0301")
        assert r.value == "\u00e9" and r.changed

    def test_normalization_can_be_disabled(self):
        assert sanitize_text("e\u0301", normalization=None).value == "e\u0301"

    @pytest.mark.parametrize("bad", [None, 5, b"x"])
    def test_non_string_rejected(self, bad):
        assert sanitize_text(bad).rejected


# ─────────────────────────────────────────────────────────────────────────────
class TestUrlBasics:
    def test_public_https_ok_and_meta_has_ips(self):
        r = validate_url("https://api.example.com/v1", resolver=public)
        assert r.ok and r.value == "https://api.example.com/v1"
        assert r.meta["host"] == "api.example.com"
        assert r.meta["resolved_ips"] == ["8.8.8.8"]

    def test_http_allowed_by_default_https_only_when_asked(self):
        assert validate_url("http://api.example.com", resolver=public).ok
        assert validate_url("http://api.example.com", allowed_schemes=("https",), resolver=public).rejected

    @pytest.mark.parametrize(
        "url",
        ["ftp://example.com/x", "file:///etc/passwd", "gopher://example.com",
         "javascript:alert(1)", "data:text/plain,hi", "//example.com/x", "example.com"],
    )
    def test_bad_schemes_rejected(self, url):
        assert validate_url(url, resolver=must_not_resolve).rejected

    def test_credentials_rejected(self):
        assert validate_url("https://user:pw@example.com", resolver=must_not_resolve).rejected
        assert validate_url("https://user@example.com", resolver=must_not_resolve).rejected

    def test_missing_host_rejected(self):
        assert validate_url("https:///path", resolver=must_not_resolve).rejected

    @pytest.mark.parametrize("url", ["https://exa mple.com", "https://example.com/\n", "https://a.com\\@b.com", "https://a.com/\x00"])
    def test_whitespace_control_and_backslash_rejected(self, url):
        # Trailing "\n" is stripped as surrounding whitespace, so only the
        # genuinely embedded cases are expected to be rejected.
        r = validate_url(url, resolver=public)
        if url == "https://example.com/\n":
            assert r.ok and r.value == "https://example.com/" and r.changed
        else:
            assert r.rejected

    def test_invalid_port_rejected(self):
        assert validate_url("https://example.com:99999", resolver=public).rejected

    def test_malformed_ipv6_rejected(self):
        assert validate_url("http://[::1", resolver=must_not_resolve).rejected

    def test_too_long_rejected(self):
        assert validate_url("https://example.com/" + "a" * 3000, resolver=public).rejected

    @pytest.mark.parametrize("bad", [None, 5, b"https://a.com"])
    def test_non_string_rejected(self, bad):
        assert validate_url(bad).rejected

    def test_surrounding_whitespace_stripped(self):
        r = validate_url("  https://example.com  ", resolver=public)
        assert r.ok and r.value == "https://example.com" and r.changed


class TestUrlSsrfLiterals:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1", "http://127.0.0.1:8000/api", "http://10.0.0.5",
            "http://192.168.1.1", "http://172.16.0.1", "http://169.254.169.254/latest/meta-data",
            "http://0.0.0.0", "http://[::1]/", "http://[fd00::1]/", "http://[fe80::1]/",
            "http://100.64.0.1", "http://255.255.255.255", "http://224.0.0.1",
        ],
    )
    def test_private_and_special_literals_blocked(self, url):
        assert validate_url(url, resolver=must_not_resolve).rejected

    @pytest.mark.parametrize(
        "url",
        ["http://2130706433/", "http://0x7f000001/", "http://0177.0.0.1/", "http://127.1/", "http://0x7f.0.0.1/"],
    )
    def test_legacy_ipv4_encodings_blocked(self, url):
        assert validate_url(url, resolver=must_not_resolve).rejected

    @pytest.mark.parametrize(
        "url",
        ["http://[::ffff:127.0.0.1]/", "http://[::ffff:7f00:1]/", "http://[::ffff:10.0.0.1]/",
         "http://[64:ff9b::7f00:1]/", "http://[2002:7f00:1::]/", "http://[::127.0.0.1]/"],
    )
    def test_ipv6_forms_embedding_ipv4_blocked(self, url):
        assert validate_url(url, resolver=must_not_resolve).rejected

    def test_public_ip_literal_allowed(self):
        assert validate_url("https://8.8.8.8/v1", resolver=must_not_resolve).ok

    def test_rejection_reason_does_not_leak_address(self):
        r = validate_url("http://10.1.2.3", resolver=must_not_resolve)
        assert "10.1.2.3" not in " ".join(r.reasons)
        assert r.meta["blocked_ip"] == "10.1.2.3"  # kept for server-side logs


class TestUrlSsrfHostnames:
    @pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "192.168.0.10", "169.254.169.254", "::1", "100.64.0.9"])
    def test_hostname_resolving_to_private_blocked(self, ip):
        assert validate_url("https://evil.example.com", resolver=lambda h: [ip]).rejected

    def test_any_private_answer_blocks_the_host(self):
        assert validate_url("https://evil.example.com", resolver=lambda h: ["8.8.8.8", "10.0.0.1"]).rejected

    @pytest.mark.parametrize(
        "url",
        ["http://localhost:8000", "http://LOCALHOST./x", "http://foo.localhost", "http://db.internal",
         "http://printer.local", "http://router.lan", "http://x.home.arpa"],
    )
    def test_local_names_blocked_without_resolving(self, url):
        assert validate_url(url, resolver=must_not_resolve).rejected

    def test_unresolvable_host_fails_closed(self):
        def boom(host):
            raise socket.gaierror("nope")
        assert validate_url("https://nx.example.com", resolver=boom).rejected

    def test_empty_resolution_fails_closed(self):
        assert validate_url("https://nx.example.com", resolver=lambda h: []).rejected

    def test_garbage_resolver_answer_rejected(self):
        assert validate_url("https://x.example.com", resolver=lambda h: ["not-an-ip"]).rejected

    def test_resolver_scope_id_is_tolerated(self):
        assert validate_url("https://x.example.com", resolver=lambda h: ["2606:4700:4700::1111%eth0"]).ok


class TestUrlAllowPrivate:
    def test_flag_allows_loopback_and_lan(self):
        assert validate_url("http://127.0.0.1:11434", allow_private=True, resolver=must_not_resolve).ok
        assert validate_url("http://192.168.1.5:8080", allow_private=True, resolver=must_not_resolve).ok
        assert validate_url("http://localhost:11434", allow_private=True, resolver=lambda h: ["127.0.0.1"]).ok

    @pytest.mark.parametrize(
        "url",
        ["http://169.254.169.254/latest", "http://169.254.170.2/", "http://100.100.100.200/",
         "http://168.63.129.16/", "http://[fd00:ec2::254]/", "http://[::ffff:169.254.169.254]/",
         "http://2852039166/"],  # decimal form of 169.254.169.254
    )
    def test_metadata_endpoints_blocked_even_when_private_allowed(self, url):
        assert validate_url(url, allow_private=True, resolver=must_not_resolve).rejected

    def test_metadata_hostnames_blocked_even_when_private_allowed(self):
        assert validate_url("http://metadata.google.internal/", allow_private=True, resolver=must_not_resolve).rejected

    def test_allow_private_still_enforces_scheme_and_credentials(self):
        assert validate_url("file:///etc/passwd", allow_private=True, resolver=must_not_resolve).rejected
        assert validate_url("http://u:p@127.0.0.1", allow_private=True, resolver=must_not_resolve).rejected

    def test_allow_private_still_blocks_unspecified_and_multicast(self):
        assert validate_url("http://0.0.0.0", allow_private=True, resolver=must_not_resolve).rejected
        assert validate_url("http://224.0.0.1", allow_private=True, resolver=must_not_resolve).rejected

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_env_flag_enables_private(self, monkeypatch, value):
        monkeypatch.setenv(ENV_ALLOW_PRIVATE_URLS, value)
        assert validate_url("http://127.0.0.1:11434", resolver=must_not_resolve).ok

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "maybe"])
    def test_env_flag_off_values_keep_blocking(self, monkeypatch, value):
        monkeypatch.setenv(ENV_ALLOW_PRIVATE_URLS, value)
        assert validate_url("http://127.0.0.1:11434", resolver=must_not_resolve).rejected

    def test_explicit_argument_beats_env(self, monkeypatch):
        monkeypatch.setenv(ENV_ALLOW_PRIVATE_URLS, "1")
        assert validate_url("http://127.0.0.1", allow_private=False, resolver=must_not_resolve).rejected

    def test_env_is_read_at_call_time(self, monkeypatch):
        assert validate_url("http://127.0.0.1", resolver=must_not_resolve).rejected
        monkeypatch.setenv(ENV_ALLOW_PRIVATE_URLS, "1")
        assert validate_url("http://127.0.0.1", resolver=must_not_resolve).ok


# ─────────────────────────────────────────────────────────────────────────────
class TestCsvInjection:
    @pytest.mark.parametrize(
        "cell",
        ["=1+1", "=cmd|' /C calc'!A0", "+cmd", "@SUM(A1:A9)", "-cmd|' /C calc'!A0", "\t=1+1", "\r=1", "@", "="],
    )
    def test_dangerous_strings_are_prefixed(self, cell):
        assert neutralize_csv_cell(cell) == "'" + cell

    @pytest.mark.parametrize("cell", ["-5", "+3.14", "-1e5", "-0.5", "hello", "a=b", "", " =x"])
    def test_safe_strings_untouched(self, cell):
        assert neutralize_csv_cell(cell) == cell

    @pytest.mark.parametrize("cell", [5, -3, 2.5, None, True, [1], {"a": 1}])
    def test_non_strings_untouched(self, cell):
        assert neutralize_csv_cell(cell) == cell

    def test_dataframe_cells_and_headers(self):
        import pandas as pd

        df = pd.DataFrame({"=evil": ["=1+1", "ok", "-7", "@x"], "n": [1, 2, 3, 4], "note": ["+cmd", "fine", None, "-2"]})
        out, n = neutralize_dataframe(df)
        assert "'=evil" in out.columns and "n" in out.columns
        assert out["'=evil"].tolist() == ["'=1+1", "ok", "-7", "'@x"]
        assert out["n"].tolist() == [1, 2, 3, 4]
        assert out["note"].tolist()[0] == "'+cmd" and out["note"].tolist()[1] == "fine"
        assert out["note"].tolist()[3] == "-2"
        assert n == 4  # 2 cells + 1 cell + 1 header

    def test_dataframe_input_not_mutated(self):
        import pandas as pd

        df = pd.DataFrame({"a": ["=1+1"]})
        neutralize_dataframe(df)
        assert df["a"].tolist() == ["=1+1"]

    def test_headers_can_be_left_alone(self):
        import pandas as pd

        out, n = neutralize_dataframe(pd.DataFrame({"=h": ["=1"]}), headers=False)
        assert "=h" in out.columns and n == 1


# ─────────────────────────────────────────────────────────────────────────────
# Property tests: the sanitizer must never crash and never emit unsafe output.
# ─────────────────────────────────────────────────────────────────────────────
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402


class TestProperties:
    @settings(max_examples=300, deadline=None)
    @given(st.text())
    def test_filename_output_is_always_safe(self, name):
        r = sanitize_filename(name, CSV_ONLY, max_length=60)
        if r.ok:
            v = r.value
            assert "/" not in v and "\\" not in v and "\x00" not in v
            assert v.lower().endswith(".csv") and len(v) <= 60
            assert not v.startswith(".") and not v.startswith(" ")
            assert v.split(".", 1)[0].strip().upper() not in {"CON", "PRN", "AUX", "NUL"}

    @settings(max_examples=300, deadline=None)
    @given(st.text())
    def test_text_output_never_contains_dangerous_chars(self, text):
        r = sanitize_text(text, max_length=500, on_too_long="truncate", allow_empty=True)
        assert r.ok
        assert "\x00" not in r.value
        assert not any(c in r.value for c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200b\ufeff")

    @settings(max_examples=300, deadline=None)
    @given(st.text())
    def test_validate_url_never_raises(self, url):
        r = validate_url(url, resolver=public)
        assert isinstance(r, SanitizeResult)

    @settings(max_examples=200, deadline=None)
    @given(st.text())
    def test_csv_cell_output_never_starts_with_formula_char_unless_number(self, s):
        out = neutralize_csv_cell(s)
        if isinstance(out, str) and out and out[0] in "=@\t\r":
            pytest.fail(f"unsafe cell survived: {out!r}")