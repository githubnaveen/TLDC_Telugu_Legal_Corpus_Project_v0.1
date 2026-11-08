import re
import unicodedata
from urllib.parse import quote

from rdflib import Namespace, URIRef

TLDC = Namespace("https://tldc.example.org/ontology#")
RESOURCE = Namespace("https://tldc.example.org/resource/")


def _key(value):
    normalized = unicodedata.normalize("NFC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", " ", normalized)


def _uri(path, value):
    key = _key(value)
    if not key:
        raise ValueError("Cannot create a graph identifier from an empty value")
    return URIRef(str(RESOURCE) + path + quote(key, safe="-_."))


def document_uri(document_id):
    return _uri("document/", document_id)


def case_uri(case_group_id):
    return _uri("case/", case_group_id)


def court_uri(court_name):
    return _uri("court/", court_name)


def legal_act_uri(act_name, canonical_id=None):
    return _uri("act/", canonical_id or act_name)


def legal_section_uri(act_name, section_identifier, canonical_act_id=None):
    act_key = _key(canonical_act_id or act_name)
    section_key = _key(section_identifier)
    if not act_key or not section_key:
        raise ValueError("Both act and section identifiers are required")
    return URIRef(str(RESOURCE) + "legal-section/" + quote(act_key, safe="-_.") + "/" + quote(section_key, safe="-_."))


def section_uri(document_id, section_id):
    return URIRef(str(document_uri(document_id)) + "/section/" + quote(str(section_id), safe="-_."))


def decision_uri(document_id):
    return URIRef(str(document_uri(document_id)) + "/decision")


def assertion_uri(assertion_key):
    return _uri("assertion/", assertion_key)
