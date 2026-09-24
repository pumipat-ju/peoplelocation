import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

test('AMBIGUOUS renders candidates and margin without declaring a match', async () => {
  const vite = await createServer({ server: { middlewareMode: true }, appType: 'custom' })
  try {
    const { AmbiguousCandidates } = await vite.ssrLoadModule('/src/EmbeddingDatabase.jsx')
    const html = renderToStaticMarkup(createElement(AmbiguousCandidates, {
      result: {
        status: 'AMBIGUOUS', margin: 0.0123, required_margin: 0.05,
        candidates: [
          { identity_session_id: 'internal-a', session_display_name: 'Session 2026-09-24 13:45',
            global_id: 7, similarity: 0.9123 },
          { identity_session_id: 'internal-b', session_display_name: 'Session 2026-09-24 14:00',
            global_id: 9, similarity: 0.9000 },
        ],
      },
    }))
    assert.match(html, /AMBIGUOUS/)
    assert.match(html, /ยังไม่สามารถยืนยันว่าเป็นบุคคลใด/)
    assert.match(html, /Session 2026-09-24 13:45/)
    assert.match(html, /Session 2026-09-24 14:00/)
    assert.match(html, /<td>7<\/td>/)
    assert.match(html, /<td>9<\/td>/)
    assert.match(html, /0\.9123/)
    assert.match(html, /0\.9000/)
    assert.match(html, /0\.0123/)
    assert.match(html, /0\.0500/)
    assert.doesNotMatch(html, /internal-a|internal-b|MATCH — พบ/)
    assert.doesNotMatch(html, /%/)
  } finally {
    await vite.close()
  }
})

test('non-ambiguous results do not render candidate panel', async () => {
  const vite = await createServer({ server: { middlewareMode: true }, appType: 'custom' })
  try {
    const { AmbiguousCandidates } = await vite.ssrLoadModule('/src/EmbeddingDatabase.jsx')
    assert.equal(renderToStaticMarkup(createElement(AmbiguousCandidates, {
      result: { status: 'MATCH', candidates: [] },
    })), '')
    assert.equal(renderToStaticMarkup(createElement(AmbiguousCandidates, {
      result: { status: 'UNKNOWN', candidates: [] },
    })), '')
  } finally {
    await vite.close()
  }
})
