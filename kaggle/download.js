// ==UserScript==
// @name         Kaggle 视频一键下载
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  美观、稳定的一键批量下载视频 (支持自由拖拽悬浮，智能识别 Output 页面)
// @author       keggin
// @match        *://www.kaggle.com/code/*
// @grant        GM_download
// ==/UserScript==

(function() {
    'use strict';
    const btn = document.createElement('button');
    btn.innerText = '一键下载视频';
    btn.style.position = 'fixed';
    btn.style.bottom = '24px';
    btn.style.right = '24px';
    btn.style.zIndex = '999999';
    btn.style.padding = '10px 20px';
    btn.style.backgroundColor = '#1F2937';
    btn.style.color = '#FFFFFF';
    btn.style.border = 'none';
    btn.style.borderRadius = '9999px';
    btn.style.cursor = 'grab';
    btn.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    btn.style.fontWeight = '500';
    btn.style.fontSize = '14px';
    btn.style.fontFamily = 'Inter, system-ui, sans-serif';
    btn.style.userSelect = 'none';
    btn.style.touchAction = 'none';
    btn.style.transition = 'background-color 0.2s ease, box-shadow 0.2s ease';
    btn.style.display = 'none';
    btn.onmouseover = () => {
        btn.style.boxShadow = '0 6px 16px rgba(0,0,0,0.25)';
    };
    btn.onmouseout = () => {
        btn.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    };

    document.body.appendChild(btn);
    let isDragging = false;
    let hasMoved = false;
    let startX = 0, startY = 0;
    let initialLeft = 0, initialTop = 0;

    const onPointerDown = (e) => {
        isDragging = true;
        hasMoved = false;

        const rect = btn.getBoundingClientRect();
        btn.style.bottom = 'auto';
        btn.style.right = 'auto';
        btn.style.left = `${rect.left}px`;
        btn.style.top = `${rect.top}px`;

        startX = e.clientX;
        startY = e.clientY;
        initialLeft = rect.left;
        initialTop = rect.top;

        btn.style.cursor = 'grabbing';

        document.addEventListener('pointermove', onPointerMove);
        document.addEventListener('pointerup', onPointerUp);
    };

    const onPointerMove = (e) => {
        if (!isDragging) return;

        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
            hasMoved = true;
        }

        let nextLeft = initialLeft + dx;
        let nextTop = initialTop + dy;

        const maxLeft = window.innerWidth - btn.offsetWidth - 8;
        const maxTop = window.innerHeight - btn.offsetHeight - 8;

        nextLeft = Math.max(8, Math.min(nextLeft, maxLeft));
        nextTop = Math.max(8, Math.min(nextTop, maxTop));

        btn.style.left = `${nextLeft}px`;
        btn.style.top = `${nextTop}px`;
    };

    const onPointerUp = () => {
        isDragging = false;
        btn.style.cursor = 'grab';
        document.removeEventListener('pointermove', onPointerMove);
        document.removeEventListener('pointerup', onPointerUp);
    };

    btn.addEventListener('pointerdown', onPointerDown);
    function checkVisibility() {
        const path = window.location.pathname;

        if (path.endsWith('/output')) {
            if (btn.style.display === 'none') {
                btn.style.display = 'block';
            }
        } else {
            if (btn.style.display === 'block') {
                btn.style.display = 'none';
            }
        }
    }

    checkVisibility();
    setInterval(checkVisibility, 500);

    // 智能获取 Script Version ID
    function getScriptVersionId() {
        const urlParams = new URLSearchParams(window.location.search);
        const idFromUrl = urlParams.get('scriptVersionId');
        if (idFromUrl) return idFromUrl;
        const links = document.querySelectorAll('a[href*="/code/out/"]');
        for (let link of links) {
            const match = link.getAttribute('href').match(/\/code\/out\/(\d+)/);
            if (match) return match[1];
        }
        try {
            const scripts = document.querySelectorAll('script');
            for (let script of scripts) {
                if (script.textContent.includes('scriptVersionId')) {
                    const match = script.textContent.match(/"scriptVersionId"\s*:\s*(\d+)/);
                    if (match) return match[1];
                }
            }
        } catch (e) {}

        return null; 
    }

    btn.addEventListener('click', (e) => {
        if (hasMoved) {
            e.preventDefault();
            return;
        }
        const scriptVersionId = getScriptVersionId();

        if (!scriptVersionId) {
            alert('抱歉，脚本无法在当前页面识别出版本 ID。\n\n请尝试刷新页面，或者随便在 Output 里点开一个文件再点击下载试试。');
            return;
        }

        const mp4Files = new Set();
        const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while ((node = walk.nextNode())) {
            const text = node.nodeValue.trim();
            if (text.endsWith('.mp4')) {
                mp4Files.add(text);
            }
        }

        const filesArray = Array.from(mp4Files);
        if (filesArray.length === 0) {
            alert('未发现 .mp4 文件，请确保 output_videos 文件夹已被展开，这样脚本才能读到文件名。');
            return;
        }

        const folderPath = prompt(`找到 ${filesArray.length} 个视频。\n请确认存放该视频的文件夹路径(留空则直接下载)：`, 'output_videos');
        if (folderPath === null) return;

        btn.innerText = '下载任务已发送';
        btn.style.backgroundColor = '#059669';

        filesArray.forEach((filename) => {
            const fullPath = folderPath ? `${folderPath}/${filename}` : filename;
            const downloadUrl = `https://www.kaggle.com/code/out/${scriptVersionId}?path=${encodeURIComponent(fullPath)}`;

            GM_download({
                url: downloadUrl,
                name: filename,
                saveAs: false
            });
        });

        setTimeout(() => {
            btn.innerText = '一键下载视频';
            btn.style.backgroundColor = '#1F2937';
        }, 3000);
    });
})();