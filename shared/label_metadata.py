def extract_questions_and_label_cols(question_answer_pairs):
    """
    Convenience wrapper to get list of questions and label_cols from a schema.
    Common starting point for analysis, iterating over questions, etc.

    Args:
        question_answer_pairs (dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}

    Returns:
        list: all questions e.g. [Question('smooth-or-featured'), ...]
        list: label_cols (list of answer strings). See ``label_metadata.py`` for for_me.
    """
    questions = list(question_answer_pairs.keys())
    label_cols = [q + answer for q, answers in question_answer_pairs.items() for answer in answers]
    return questions, label_cols


gz2_pairs = {
    'smooth_or_featured': ['_smooth', '_featured_or_disk', '_artifact'],
    'disk_edge_on': ['_yes', '_no'],
    'has_spiral_arms': ['_yes', '_no'],
    'bar': ['_strong', '_weak', '_no'],
    'bulge_size': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how_rounded': ['_round', '_in_between', '_cigar_shaped'],
    'edge_on_bulge': ['_boxy', '_none', '_rounded'], # de
    'spiral_winding': ['_tight', '_medium', '_loose'],
    'spiral_arm_count': ['_1', '_2', '_3', '_4', '_more_than_4', '_cant_tell'],
    'merging': ['_none', '_minor_disturbance', '_major_disturbance', '_merger'] # de
}

gz2_questions, gz2_label_cols = extract_questions_and_label_cols(gz2_pairs)


gz2_and_decals_dependencies = {
    'smooth_or_featured': None,  # always asked
    'disk_edge_on': 'smooth_or_featured_featured_or_disk',
    'has_spiral_arms': 'smooth_or_featured_featured_or_disk',
    'bar': 'smooth_or_featured_featured_or_disk',
    'bulge_size': 'smooth_or_featured_featured_or_disk',
    'how_rounded': 'smooth_or_featured_smooth',
    'edge_on_bulge': 'disk_edge_on_yes',
    'spiral_winding': 'has_spiral_arms_yes',
    'spiral_arm_count': 'has_spiral_arms_yes', # bad naming...
    'merging': None,

}

