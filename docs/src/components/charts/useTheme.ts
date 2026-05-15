import { useState, useEffect } from 'react';

export function useTheme() {
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    const html = document.documentElement;
    const update = () => setIsDark(html.dataset.theme !== 'light');
    update();

    const observer = new MutationObserver(update);
    observer.observe(html, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  return isDark;
}

export function themeColors(isDark: boolean) {
  return {
    text: isDark ? '#a1a1aa' : '#52525b',
    grid: isDark ? '#27272a' : '#e4e4e7',
    tooltip: {
      bg: isDark ? '#18181b' : '#faf8f3',
      border: isDark ? '#3f3f46' : '#e4e4e7',
      text: isDark ? '#fafafa' : '#18181b',
    },
  };
}
