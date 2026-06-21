from io import StringIO

from yolo_exploration.utils.text_stream import (
    set_text_stream_filter,
    unset_text_stream_filter,
)


def test_filtered_text_stream_substitutes_multiple_strings() -> None:
    """
    Apply all substitutions to text written in one call.
    """
    target = StringIO()
    stream = set_text_stream_filter(
        target,
        map={"secret": "public", "/project/": "workspace/"},
    )
    text = "secret: /project/data/image.jpg"

    written = stream.write(text)

    assert written == len(text)
    assert target.getvalue() == "public: workspace/data/image.jpg"


def test_filtered_text_stream_filters_each_write_independently() -> None:
    """
    Do not match strings split across separate writes.
    """
    target = StringIO()
    stream = set_text_stream_filter(target, map={"secret": "public"})

    stream.write("sec")
    stream.write("ret")

    assert target.getvalue() == "secret"


def test_filtered_text_stream_filters_writelines() -> None:
    """
    Apply substitutions to every line passed to writelines.
    """
    target = StringIO()
    stream = set_text_stream_filter(target, map={"private/": "public/"})

    stream.writelines(("private/one\n", "private/two\n"))

    assert target.getvalue() == "public/one\npublic/two\n"


def test_filtered_text_stream_delegates_other_attributes() -> None:
    """
    Delegate attributes not implemented by the filter to its target.
    """
    target = StringIO()
    stream = set_text_stream_filter(target, map={})

    assert stream.seekable() == target.seekable()


def test_unset_text_stream_filter_returns_original_stream() -> None:
    """
    Return the original target when unsetting a filter.
    """
    target = StringIO()
    stream = set_text_stream_filter(target, map={"secret": "public"})

    unfiltered_stream = unset_text_stream_filter(stream)
    unfiltered_stream.write("secret")

    assert unfiltered_stream is target
    assert target.getvalue() == "secret"


def test_unset_text_stream_filter_leaves_plain_stream_unchanged() -> None:
    """
    Return a stream without a filter unchanged.
    """
    target = StringIO()

    unfiltered_stream = unset_text_stream_filter(target)

    assert unfiltered_stream is target


def test_setting_filter_again_does_not_stack_filters() -> None:
    """
    Replace an existing filter instead of stacking another wrapper.
    """
    target = StringIO()
    stream = set_text_stream_filter(target, map={"old": "first"})

    replacement_stream = set_text_stream_filter(stream, map={"new": "second"})

    assert unset_text_stream_filter(replacement_stream) is target


def test_filtered_text_stream_replaces_a_different_filter() -> None:
    """
    Apply only the substitutions from the replacement filter.
    """
    target = StringIO()
    stream = set_text_stream_filter(target, map={"old": "first"})

    replacement_stream = set_text_stream_filter(stream, map={"new": "second"})
    replacement_stream.write("old and new")

    assert target.getvalue() == "old and second"
