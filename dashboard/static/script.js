/**
 * Клиентский скрипт для обновления дашборда
 */

class DashboardClient {
    constructor() {
        this.ws = null;
        this.updateInterval = null;
        this.chart = null;
        this.lastUpdateTime = null;

        this.init();
    }

    init() {
        // Инициализация WebSocket соединения
        this.connectWebSocket();

        // Инициализация графика
        this.initChart();

        // Запуск периодического обновления (fallback)
        this.startPolling();

        // Настройка обработчиков событий
        this.setupEventListeners();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateDashboard(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('disconnected');

            // Попытка переподключения через 5 секунд
            setTimeout(() => this.connectWebSocket(), 5000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('error');
        };
    }

    initChart() {
        const ctx = document.getElementById('discussionChart').getContext('2d');

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Topic Activity',
                        data: [],
                        borderColor: '#2d8cff',
                        backgroundColor: 'rgba(45, 140, 255, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Entities Mentioned',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            display: true
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    startPolling() {
        // Fallback polling каждые 30 секунд если WebSocket не работает
        this.updateInterval = setInterval(() => {
            if (this.ws.readyState !== WebSocket.OPEN) {
                this.fetchData();
            }
        }, 30000);

        // Первоначальная загрузка данных
        this.fetchData();
    }

    async fetchData() {
        try {
            const response = await fetch('/api/state');
            const data = await response.json();
            this.updateDashboard(data);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    updateDashboard(data) {
        this.lastUpdateTime = new Date();

        // Обновляем статус
        if (data.meeting_status) {
            this.updateStatus(data.meeting_status);
        }

        // Обновляем текущую тему
        this.updateCurrentTopic(data.current_topic);

        // Обновляем справку
        this.updateSummary(data.summary);

        // Обновляем сущности
        this.updateEntities(data.entities);

        // Обновляем решения
        this.updateDecisions(data.decisions);

        // Обновляем вопросы
        this.updateQuestions(data.open_questions);

        // Обновляем новости
        this.updateNews(data.news);

        // Обновляем статистику
        this.updateStats(data.stats);

        // Обновляем график
        this.updateChart(data.discussion_timeline);

        // Обновляем время последнего обновления
        this.updateLastUpdated();
    }

    updateStatus(status) {
        const indicator = document.getElementById('statusIndicator');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');

        dot.className = 'status-dot';
        dot.classList.add(status);

        const statusTexts = {
            'connecting': 'Connecting...',
            'connected': 'Connected to Meeting',
            'in_meeting': 'In Meeting',
            'screen_sharing': 'Screen Sharing Active',
            'disconnected': 'Disconnected',
            'error': 'Connection Error'
        };

        text.textContent = statusTexts[status] || status;
    }

    updateCurrentTopic(topic) {
        const container = document.getElementById('currentTopic');

        if (!topic || topic === 'No topic detected') {
            container.innerHTML = '<p class="topic-placeholder">Waiting for discussion to start...</p>';
            return;
        }

        container.innerHTML = `
            <p class="topic-text">${topic}</p>
            ${topic.subtopics ? `
                <div class="subtopics">
                    ${topic.subtopics.map(st => `<span class="subtopic-tag">${st}</span>`).join('')}
                </div>
            ` : ''}
        `;
    }

    updateSummary(summary) {
        const container = document.getElementById('summaryContent');

        if (!summary || !summary.length) {
            container.innerHTML = `
                <ul class="summary-list">
                    <li>Discussion summary will appear here</li>
                    <li>Key arguments and context</li>
                    <li>Main points being discussed</li>
                </ul>
            `;
            return;
        }

        const items = summary.map(item => `<li>${item}</li>`).join('');
        container.innerHTML = `<ul class="summary-list">${items}</ul>`;
    }

    updateEntities(entities) {
        const container = document.getElementById('entitiesContainer');

        if (!entities || !entities.length) {
            container.innerHTML = `
                <div class="entity-tags">
                    <span class="entity-tag placeholder">No entities detected yet</span>
                </div>
            `;
            return;
        }

        const tags = entities.map(entity => {
            const typeClass = entity.type ? ` ${entity.type}` : '';
            return `<span class="entity-tag${typeClass}">${entity.name}</span>`;
        }).join('');

        container.innerHTML = `<div class="entity-tags">${tags}</div>`;
    }

    updateDecisions(decisions) {
        const container = document.getElementById('decisionsList');

        if (!decisions || !decisions.length) {
            container.innerHTML = `
                <div class="decision-item placeholder">
                    <span class="decision-checkbox"></span>
                    <span class="decision-text">Agreements and action items will appear here</span>
                </div>
            `;
            return;
        }

        const items = decisions.map((decision, index) => `
            <div class="decision-item" data-id="${index}">
                <span class="decision-checkbox ${decision.completed ? 'checked' : ''}"></span>
                <span class="decision-text">${decision.text}</span>
                ${decision.assignee ? `<span class="decision-assignee">${decision.assignee}</span>` : ''}
            </div>
        `).join('');

        container.innerHTML = items;
    }

    updateQuestions(questions) {
        const container = document.getElementById('questionsList');

        if (!questions || !questions.length) {
            container.innerHTML = `
                <div class="question-item placeholder">
                    <span class="question-icon">?</span>
                    <span class="question-text">Unresolved questions will appear here</span>
                </div>
            `;
            return;
        }

        const items = questions.map((question, index) => `
            <div class="question-item" data-id="${index}">
                <span class="question-icon">?</span>
                <span class="question-text">${question}</span>
            </div>
        `).join('');

        container.innerHTML = items;
    }

    updateNews(newsItems) {
        const container = document.getElementById('newsList');

        if (!newsItems || !newsItems.length) {
            container.innerHTML = `
                <div class="news-item placeholder">
                    <div class="news-source">News related to mentioned entities will appear here</div>
                    <div class="news-summary">With summaries and links to sources</div>
                </div>
            `;
            return;
        }

        const items = newsItems.map(item => `
            <div class="news-item">
                <div class="news-source">
                    ${item.source} • ${item.date}
                    ${item.freshness ? `<span class="news-fresh-badge">${item.freshness}</span>` : ''}
                </div>
                <div class="news-summary">${item.summary}</div>
                ${item.url ? `<a href="${item.url}" target="_blank" class="news-link">Read more →</a>` : ''}
            </div>
        `).join('');

        container.innerHTML = items;
    }

    updateStats(stats) {
        if (!stats) return;

        if (stats.word_count) {
            document.getElementById('wordCount').textContent = `Words: ${stats.word_count}`;
        }

        if (stats.speaker_count) {
            document.getElementById('speakerCount').textContent = `Speakers: ${stats.speaker_count}`;
        }

        if (stats.duration) {
            document.getElementById('duration').textContent = `Duration: ${stats.duration}`;
        }
    }

    updateChart(timelineData) {
        if (!timelineData || !this.chart) return;

        if (timelineData.labels && timelineData.datasets) {
            this.chart.data.labels = timelineData.labels;
            this.chart.data.datasets = timelineData.datasets;
            this.chart.update();
        }
    }

    updateLastUpdated() {
        const element = document.getElementById('lastUpdated');
        if (this.lastUpdateTime) {
            const timeStr = this.lastUpdateTime.toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            element.textContent = `Last updated: ${timeStr}`;
        }
    }

    setupEventListeners() {
        // Обработчик для кнопки обновления новостей
        window.refreshNews = async () => {
            try {
                const response = await fetch('/api/news/refresh', { method: 'POST' });
                const data = await response.json();
                this.updateNews(data.news);
            } catch (error) {
                console.error('Error refreshing news:', error);
            }
        };

        // Обработчики для чекбоксов решений
        document.addEventListener('click', (e) => {
            if (e.target.closest('.decision-checkbox')) {
                const decisionItem = e.target.closest('.decision-item');
                const decisionId = decisionItem.dataset.id;
                this.toggleDecision(decisionId);
            }
        });
    }

    async toggleDecision(decisionId) {
        try {
            const response = await fetch(`/api/decisions/${decisionId}/toggle`, {
                method: 'POST'
            });
            const data = await response.json();

            if (data.success) {
                const checkbox = document.querySelector(`[data-id="${decisionId}"] .decision-checkbox`);
                checkbox.classList.toggle('checked');
            }
        } catch (error) {
            console.error('Error toggling decision:', error);
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardClient = new DashboardClient();
});