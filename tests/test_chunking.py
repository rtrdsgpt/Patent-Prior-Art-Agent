from patent_agent.ingestion.chunking import claim_to_index_chunk, split_claims

SAMPLE_CLAIMS = """1. A method for training a neural network, comprising:
receiving a set of labeled training examples; and
updating a plurality of weights of the neural network using backpropagation.

2. The method of claim 1, wherein the backpropagation comprises computing a gradient of a loss function with respect to each of the plurality of weights.

3. The method of claim 1, further comprising applying a dropout regularization technique during training.

4. A system comprising:
a memory storing a neural network; and
a processor configured to update a plurality of weights of the neural network using backpropagation.

5. The system of claims 1-4, wherein the neural network is a convolutional neural network.
"""


def test_split_claims_returns_correct_count():
    claims = split_claims(SAMPLE_CLAIMS)
    assert len(claims) == 5


def test_split_claims_assigns_sequential_numbers():
    claims = split_claims(SAMPLE_CLAIMS)
    assert [c.claim_number for c in claims] == [1, 2, 3, 4, 5]


def test_split_claims_identifies_independent_claims():
    claims = split_claims(SAMPLE_CLAIMS)
    by_number = {c.claim_number: c for c in claims}
    assert by_number[1].is_independent is True
    assert by_number[4].is_independent is True


def test_split_claims_identifies_dependent_claims_and_target():
    claims = split_claims(SAMPLE_CLAIMS)
    by_number = {c.claim_number: c for c in claims}
    assert by_number[2].is_independent is False
    assert by_number[2].depends_on == 1
    assert by_number[3].depends_on == 1


def test_split_claims_dependency_reference_inside_claim_body_does_not_create_extra_claim():
    # Claims 2/3/5's bodies reference "claim 1" mid-sentence, which must not be parsed as
    # a new claim boundary (boundary matches require the number to start a line).
    claims = split_claims(SAMPLE_CLAIMS)
    assert {c.claim_number for c in claims} == {1, 2, 3, 4, 5}


def test_split_claims_handles_multi_claim_dependency_takes_first_reference():
    claims = split_claims(SAMPLE_CLAIMS)
    by_number = {c.claim_number: c for c in claims}
    assert by_number[5].depends_on == 1


def test_split_claims_strips_claim_text_whitespace():
    claims = split_claims(SAMPLE_CLAIMS)
    assert not claims[0].text.startswith("\n")
    assert not claims[0].text.endswith("\n")


def test_split_claims_empty_input_returns_empty_list():
    assert split_claims("") == []


def test_split_claims_ignores_out_of_sequence_numbers():
    # A stray "7. " inside body text (e.g. a sub-list) must not be treated as a new claim
    # since it isn't the expected next sequential number.
    text = "1. A method comprising step 7. further processing.\n\n2. The method of claim 1."
    claims = split_claims(text)
    assert [c.claim_number for c in claims] == [1, 2]


def test_claim_to_index_chunk_includes_title_and_claim_number():
    claims = split_claims(SAMPLE_CLAIMS)
    chunk = claim_to_index_chunk(claims[0], patent_title="Neural network training method")
    assert "Neural network training method" in chunk
    assert "Claim 1" in chunk
    assert "backpropagation" in chunk
