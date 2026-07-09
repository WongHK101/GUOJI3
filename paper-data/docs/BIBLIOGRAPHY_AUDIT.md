# Bibliography Audit: Route-B Draft v2

**Audit date:** 2026-07-10  
**Input:** The 40 `\bibitem` entries embedded in `istf_kbs_jacobian_coverage_draft_v2.tex`.  
**Method:** Read-only Crossref title resolution followed by exact-title comparison. A low-confidence or wrong Crossref candidate is not treated as verification. No reference was added, deleted, or silently corrected in this audit.

## Summary

- 17 entries have an exact-title Crossref resolution and matching venue/year information sufficient for the fields used in the current draft.
- 23 entries are retained but marked `MANUAL_OFFICIAL_CHECK_REQUIRED`, either because they are conference proceedings/books or because Crossref returned an ambiguous candidate. They are not supplemented with guessed DOI values.
- Before submission, every `MANUAL_OFFICIAL_CHECK_REQUIRED` row must be resolved against the listed official publisher, proceedings, or book-catalog source. The manuscript must not gain any new citation until its metadata is verified.

## Crossref-verified entries

| Citation key | DOI / official record | Fields checked |
| --- | --- | --- |
| granger1969 | `10.2307/1912791` | Title, year, journal, volume, issue, pages; author list requires final publisher comparison. |
| nauta2019 | `10.3390/make1010019` | Title, year, journal, volume, issue, article number; author list requires final publisher comparison. |
| cheng2024cutsplus | `10.1609/aaai.v38i10.29034` | Title, year, proceedings venue, volume, issue, pages; author list requires final proceedings comparison. |
| zhang2017cdnod | `10.24963/ijcai.2017/187` | Title, year, proceedings venue, pages; author list requires final proceedings comparison. |
| perez2018 | `10.1609/aaai.v32i1.11671` | Title, year, proceedings venue, volume, issue; author list requires final proceedings comparison. |
| lim2021 | `10.1016/j.ijforecast.2021.03.012` | Title, year, journal, volume, issue, pages; author list requires final publisher comparison. |
| torralba2011 | `10.1109/CVPR.2011.5995347` | Title, year, proceedings venue, pages; author list requires final proceedings comparison. |
| ross2017right | `10.24963/ijcai.2017/371` | Title, year, proceedings venue, pages; author list requires final proceedings comparison. |
| vowels2023 | `10.1145/3527154` | Title, year, journal, volume, issue, article number; author list requires final publisher comparison. |
| runge2018 | `10.1063/1.5025050` | Title, year, journal, volume, issue, article number; author list requires final publisher comparison. |
| assaad2022 | `10.1613/jair.1.13428` | Title, year, journal, volume, article number; author list requires final publisher comparison. |
| gong2024 | `10.1145/3705297` | Title, year, journal, volume, issue, article number; author list requires final publisher comparison. |
| yang2024kbs | `10.1016/j.knosys.2024.111865` | Title, year, Knowledge-Based Systems volume/article number; author list requires final publisher comparison. |
| chen2024kbs | `10.1016/j.knosys.2024.111868` | Title, year, Knowledge-Based Systems volume/article number; author list requires final publisher comparison. |
| sun2025pcac | `10.1016/j.knosys.2025.114135` | Title, year, Knowledge-Based Systems volume/article number; author list requires final publisher comparison. |
| tao2025active | `10.1016/j.knosys.2025.114145` | Title, year, Knowledge-Based Systems volume/article number; author list requires final publisher comparison. |
| lutkepohl2005 | `10.1007/3-540-27752-8` | Book title, year, publisher record; author requires final publisher comparison. |

## Manual official-source checks required

| Citation key | Official source to use | Fields currently checked | Reason for manual follow-up |
| --- | --- | --- | --- |
| geweke1982 | Journal of the American Statistical Association publisher/JSTOR record | Title, authors, year, journal, volume, issue, pages pending official match. | Crossref title query returned a rejoinder rather than the cited article. |
| tank2021 | IEEE Xplore / IEEE TPAMI record | Title, authors, journal, volume, issue, pages pending official match. | Crossref title query was ambiguous. |
| khanna2021 | NeurIPS proceedings record | Title, authors, year, proceedings metadata pending. | Proceedings item needs official verification. |
| jrngc2024 | ICML/PMLR or stated official proceedings record | Title, authors, year, venue metadata pending. | Crossref candidate was a different paper. |
| lowe2022acd | CLeaR/PMLR proceedings record | Title, authors, year, volume, pages pending. | Crossref candidate was ambiguous. |
| gong2023rhino | ICLR/OpenReview proceedings record | Title, authors, year, venue pending. | Crossref candidate was unrelated. |
| runge2019 | Science Advances publisher record | Title, authors, year, journal, volume, issue/article number pending. | Crossref query returned a recommendation record. |
| runge2020 | UAI/PMLR proceedings record | Title, authors, year, volume, pages pending. | Crossref query was ambiguous. |
| runge2023 | Nature Reviews Earth & Environment publisher record | Title, authors, year, journal, volume, pages pending. | Crossref candidate was a book chapter. |
| peters2013 | NeurIPS proceedings record | Title, authors, year, proceedings pages pending. | Crossref candidate was unrelated. |
| schreiber2000 | Physical Review Letters publisher record | Title, author, year, journal, volume, issue, pages pending. | Crossref candidate was unrelated. |
| pamfil2020 | AISTATS/PMLR proceedings record | Title, authors, year, volume, pages pending. | Crossref candidate was unrelated. |
| bai2018 | arXiv record | Title, authors, year, arXiv identifier pending. | Preprint requires canonical arXiv metadata check. |
| vaswani2017 | NeurIPS proceedings record | Title, authors, year, proceedings metadata pending. | Crossref candidate was a different work. |
| gu2022 | ICLR/OpenReview proceedings record | Title, authors, year, proceedings metadata pending. | Crossref candidate was unrelated. |
| gu2023 | NeurIPS proceedings record | Title, authors, year, proceedings metadata pending. | Crossref candidate was a different work. |
| geirhos2020 | Nature Machine Intelligence publisher record | Title, authors, year, journal, volume, pages pending. | Crossref candidate was unrelated. |
| ilyas2019adversarial | NeurIPS proceedings record | Title, authors, year, proceedings metadata pending. | Crossref candidate was a discussion article. |
| scholkopf2021 | Proceedings of the IEEE publisher record | Title, authors, year, journal, volume, issue, pages pending. | Crossref candidate was unrelated. |
| liu2024qfm | Stated publisher/proceedings record | Title, authors, year, venue, volume/article number pending. | Crossref candidate was an SSRN record. |
| pearl2009 | Cambridge University Press catalog | Book title, edition, author, year pending official catalog check. | Crossref returned a review of an earlier edition. |
| spirtes2000 | MIT Press catalog | Book title, edition, authors, year pending official catalog check. | Crossref year did not match the cited edition. |
| peters2017 | MIT Press catalog | Book title, authors, year pending official catalog check. | Crossref returned a book review rather than the book record. |

## Editorial rule for the next drafting pass

Keep the current citations exactly as they are until a verified metadata update is prepared. Any corrected volume, issue, page, article number, or DOI must be applied consistently in the TeX bibliography and recorded in a follow-up audit, rather than being changed ad hoc during prose revision.
