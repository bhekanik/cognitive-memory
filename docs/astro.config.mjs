// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

import react from '@astrojs/react';

export default defineConfig({
  site: 'https://planetaryescape.github.io',
  base: '/cognitive-memory',
  integrations: [starlight({
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
    customCss: ['./src/styles/custom.css'],
    components: {
      Head: './src/components/Head.astro',
      Footer: './src/components/Footer.astro',
      Hero: './src/components/Hero.astro',
    },
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/planetaryescape/cognitive-memory' }],
    editLink: { baseUrl: 'https://github.com/planetaryescape/cognitive-memory/edit/main/docs/' },
  }), react()],
});
