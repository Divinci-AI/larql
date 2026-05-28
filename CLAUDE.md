# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reference the `./AGENTS.md` file for the broader context of this project.

## Fork relationship

This repo (`Divinci-AI/larql`) is a fork of [`chrishayuk/larql`](https://github.com/chrishayuk/larql). The upstream remote is configured as `upstream`. The fork's reason to exist is the RFC-0001 mechanistic fact-editing surface (`crown` / `edit` / `apply-patch` / `memit` + PyO3 bindings) plus the fp8-block-quant decode for Kimi-K2 / DeepSeek-V3 — everything else should track upstream tightly.

When the user mentions **"upstream"**, **"harvest"**, **"sync with chrishayuk"**, **"merge upstream"**, or **"RFC-000N harvest"**, the `larql-upstream-harvest` skill (`.claude/skills/larql-upstream-harvest/SKILL.md`) is the playbook — survey-first triage, reset-and-replay decision tree, our Commands/DevCommand conflict patterns, fp8-block-quant preservation, force-push warning, and the RFC-as-living-document convention from RFC-0002.

## Project RFC index

- `docs/rfcs/0001-mechanistic-fact-editing.md` — the fork's product identity (crown/edit/memit).
- `docs/rfcs/0002-upstream-harvest-2026-05-28.md` — worked example of a large harvest with the pivot-mid-stream pattern (plan + Wave 1 log + Wave 2 log).