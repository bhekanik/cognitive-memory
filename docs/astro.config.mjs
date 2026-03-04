// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Cognitive Memory',
      description: 'Biologically-inspired agent memory with decay, consolidation, and tiered storage',
      sidebar: [
        { slug: 'index' },
        { label: 'Getting Started', autogenerate: { directory: 'getting-started' } },
        { label: 'Concepts', autogenerate: { directory: 'concepts' } },
        { label: 'Adapters', autogenerate: { directory: 'adapters' } },
        { label: 'Benchmarks', autogenerate: { directory: 'benchmarks' } },
        { label: 'API Reference', autogenerate: { directory: 'api' } },
        { label: 'Guides', autogenerate: { directory: 'guides' } },
      ],
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/bhekanik/cognitive-memory' }],
      editLink: { baseUrl: 'https://github.com/bhekanik/cognitive-memory/edit/main/docs/' },
    }),
  ],
});
