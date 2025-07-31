# -------------------------
# app.py (sidebar navigation version)
# -------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import numpy as np
import seaborn as sns
import random
from datetime import datetime

from matplotlib.patches import Patch
import matplotlib.patches as mpatches


# -------------------------
# 0) Page Config (must be first Streamlit call)
# -------------------------

st.set_page_config(
    page_title="Customer's Behavior Analysis",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# 2) Title + Divider
# -------------------------
col1, col2 = st.columns([1,19])  # Adjust ratio

with col1:
    st.image("logo.jpg", width=60)  # Put your image path and size here

with col2:
    st.title("Customer's Behavior Analysis")
st.markdown("<hr style='border: 3px solid white; margin-top: -10px; margin-bottom: 20px;'>",
            unsafe_allow_html=True)


# -------------------------
# 1) Global Styles (CSS)
# -------------------------
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f3f6fa;
        padding: 1rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.03);
    }
    div[data-testid="metric-container"] > div {
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# 3) Data
# -------------------------
df = pd.read_csv("ecommerce_customer_behavior_dataset.csv")

# helpers for table styling
def random_color():
    return f'background-color: #{random.randint(0, 0xFFFFFF):06x}'

def style_cells_random(df_):
    return df_.style.applymap(lambda _: random_color())

# mappings / engineered cols used later
satisfaction_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
if 'Customer Satisfaction' in df.columns:
    df['Customer Satisfaction Numeric'] = df['Customer Satisfaction'].map(satisfaction_mapping)

# -------------------------
# 4) Sidebar: Date, Theme, Sections, Metrics, Profile
# -------------------------

now = datetime.now()
formatted_now = now.strftime("%d %B %Y, %A")
st.sidebar.title(formatted_now)

st.sidebar.divider()  # Adds a line + space



# ------------------ THEME: Dark / Light toggle ------------------
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.get("dark_mode", False))
    st.session_state["dark_mode"] = dark_mode

# Apply plotting templates/styles
def apply_theme(dark: bool):
    # Plotly
    px.defaults.template = "plotly_dark" if dark else "plotly_white"

    # Matplotlib / Seaborn
    mpl.rcParams.update({
        "figure.facecolor": "#0E1117" if dark else "white",
        "axes.facecolor": "#0E1117" if dark else "white",
        "axes.edgecolor": "white" if dark else "black",
        "text.color": "white" if dark else "black",
        "axes.labelcolor": "white" if dark else "black",
        "xtick.color": "white" if dark else "black",
        "ytick.color": "white" if dark else "black",
        "grid.color": "#444" if dark else "#ccc",
    })
    sns.set_theme(style="darkgrid" if dark else "whitegrid")

apply_theme(dark_mode)

# A little CSS polish for background/text (optional)
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 3rem;
        padding-bottom: 0.5rem;
    }}

    html, body, [class^="css"]  {{
        background-color: {"#0E1117" if dark_mode else "white"} !important;
        color: {"#FAFAFA" if dark_mode else "#111"} !important;
    }}

    .kpi-card {{
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        background: {"#1A1F2B" if dark_mode else "#f2f4ff"};
        border: 1px solid {"#2B3244" if dark_mode else "#c5cbe8"};
    }}

    /* Make the KPI title more visible in light mode */
    .kpi-title {{
        font-size: 13px;
        margin-bottom: 4px;
        font-weight: {"600" if dark_mode else "700"};     /* bolder in light mode */
        opacity: {"0.9" if dark_mode else "1"};           /* no transparency in light mode */
        color: {"#E6E8F0" if dark_mode else "#0B132B"};   /* darker text in light mode */
        letter-spacing: {"0" if dark_mode else "0.3px"};  /* subtle emphasis in light mode */
        text-transform: {"none" if dark_mode else "uppercase"}; /* optional: stronger presence in light */
    }}

    .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        line-height: 1.1;
        color: {"#FAFAFA" if dark_mode else "#111"};
    }}
    </style>
    """,
    unsafe_allow_html=True
)




# Map Customer Satisfaction Numeric if not present
if 'Customer Satisfaction Numeric' not in df.columns and 'Customer Satisfaction' in df.columns:
    satisfaction_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Customer Satisfaction Numeric'] = df['Customer Satisfaction'].map(satisfaction_mapping)

# ------------------ KPI CARDS ------------------
def kpi(col, title, value):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Compute KPIs safely
total_customers = len(df)
return_rate = (df['Return Customer'].mean() * 100) if 'Return Customer' in df.columns and total_customers else 0
avg_purchase = df['Purchase Amount ($)'].mean() if 'Purchase Amount ($)' in df.columns else 0
total_revenue = df['Purchase Amount ($)'].sum() if 'Purchase Amount ($)' in df.columns else 0
avg_review = df['Review Score (1-5)'].mean() if 'Review Score (1-5)' in df.columns else 0
avg_satisfaction = df['Customer Satisfaction Numeric'].mean() if 'Customer Satisfaction Numeric' in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "👥 Total Customers", f"{total_customers:,}")
kpi(c2, "🔁 Return Rate", f"{return_rate:,.2f}%")
kpi(c3, "💵 Avg Purchase", f"${avg_purchase:,.2f}")
kpi(c4, "🧾 Total Revenue", f"${total_revenue:,.2f}")
kpi(c5, "⭐ Avg Review / Sat.", f"{avg_review:,.2f} / {avg_satisfaction:,.2f}")

st.markdown("---")

st.sidebar.divider()  # Adds a line + space


# -------------------------
# 5) Sidebar Navigation (replaces tabs)
# -------------------------
PAGES = [
    "📊 Age Stats",
    "💰 Purchase Amount Stats",
    "📱 Device Type Stats",
    "🛍️ Product Category Stats",
    "🎟️ Discount Analysis",
    "✅ Satisfied Return Customers",
    "🕒 Time Spent vs Purchase"
]
st.sidebar.markdown("## 📂 Sections")
page = st.sidebar.radio(label="", options=PAGES, index=0, key="nav_choice")

# -------------------------
# 6) Page Implementations
# -------------------------

def page_age_stats(df: pd.DataFrame):
    st.markdown("<h2 style='text-align: left;'>Age</h2>", unsafe_allow_html=True)

    mean = round(df['Age'].mean(), 2)
    median = round(df['Age'].median(), 2)
    mode = round(df['Age'].mode()[0], 2)

    age_stats = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Mode"],
        "Value": [mean, median, mode],
    })


    col_table, col_checkbox = st.columns([3, 1])
    with col_table:
        st.dataframe(
            style_cells_random(age_stats).format({"Value": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

    with col_checkbox:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                    unsafe_allow_html=True)
        show_pie = st.checkbox(label='Show Pie Chart')

    # NEW ROW: put the chart below, half-width
    if show_pie:
        values = [mean, median, mode]
        labels = ['Mean', 'Median', 'Mode']
        fig, ax = plt.subplots(figsize=(9, 8))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(values, labels=labels, autopct='%1.2f%%', startangle=90,
               colors=colors, pctdistance=0.5, shadow=True)
        ax.set_title('Age Distribution of Users', fontsize=14)
        ax.axis('equal')

        col_left, col_right = st.columns([1, 1])  # half page width
        with col_left:
            st.pyplot(fig, use_container_width=True)
        # col_right left empty intentionally

def page_purchase_amount_stats(df: pd.DataFrame):
    st.markdown("<h2 style='text-align: left;'>Purchase Amount</h2>", unsafe_allow_html=True)
    variance = round(df['Purchase Amount ($)'].var(), 2)
    std = round(df['Purchase Amount ($)'].std(), 2)
    iqr = round(df['Purchase Amount ($)'].quantile(0.75) - df['Purchase Amount ($)'].quantile(0.25), 2)

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric(label='Variance', value=f"{variance:.2f}")
    col2.metric(label='Standard Deviation', value=f"{std:.2f}")
    col3.metric(label='Interquartile Range', value=f"{iqr:.2f}")

def page_device_type_stats(df: pd.DataFrame):
    st.markdown("<h2 style='text-align: left;'>Percentages of Devices Users</h2>", unsafe_allow_html=True)

    devices = df['Device Type'].value_counts()
    devices_percentages = (devices / devices.sum()) * 100
    device_list = pd.Series(devices_percentages.index)
    percentage = pd.Series(devices_percentages.values)

    col9, col10, col11, col12 = st.columns([1, 1, 1, 1])
    if len(device_list) >= 1:
        col9.metric(label=device_list[0], value=f"{percentage[0]:.2f}%")
    if len(device_list) >= 2:
        col10.metric(label=device_list[1], value=f"{percentage[1]:.2f}%")
    if len(device_list) >= 3:
        col11.metric(label=device_list[2], value=f"{percentage[2]:.2f}%")

    with col12:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                    unsafe_allow_html=True)
        show_pie = st.checkbox(label="Show Pie Visualization", key='pie_box2')

    # 🔽 NEW ROW: render the chart below, half-width on the left
    if show_pie:
        colors = ['#66b3ff', '#99ff99', '#ffcc99']  # adjust if you have >3 device types
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            percentage,
            labels=device_list,
            colors=colors[:len(device_list)],
            autopct='%.1f%%',
            startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        ax.set_title('Percentages of Customers by Device Type', fontsize=14, fontweight='bold')

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.pyplot(fig, use_container_width=True)
        # col_right intentionally left empty

def page_product_category_stats(df: pd.DataFrame):
    st.markdown("<h2 style='text-align: left;'>🛍️ Product Category Stats</h2>", unsafe_allow_html=True)

    grouped = df.groupby('Product Category')['Purchase Amount ($)'].sum().sort_values(ascending=False)
    product_category_df = grouped.reset_index()
    product_category_df.columns = ['Product Category', 'Total Purchase Amount ($)']

    # Table + controls
    col6, col_spacer, col7 = st.columns([2.5, 0.2, 1.3])
    with col6:
        st.markdown("#### 🎯 Product Category Purchase Summary")
        st.dataframe(
            style_cells_random(product_category_df)
            .format({'Total Purchase Amount ($)': '${:,.2f}'}),
            use_container_width=True,
            hide_index=True
        )

    with col7:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                    unsafe_allow_html=True)
        show_bar = st.checkbox("Show Bar Chart", key='bar_box')
        show_pie = st.checkbox("Show Pie Chart (Top 3)", key='pie_box1')

    # One row for both plots
    if show_bar or show_pie:
        col_left, col_right = st.columns([1, 1])

        # --- Bar chart ---
        if show_bar:
            fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
            sns.barplot(x=grouped.index, y=grouped.values, palette='Set3', edgecolor='black', ax=ax_bar)
            sns.scatterplot(x=grouped.index, y=grouped.values, color='black', s=100, zorder=10, ax=ax_bar)
            for i, value in enumerate(grouped.values):
                ax_bar.text(i, value + 10, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            ax_bar.set_title('Average Purchase Amount by Product Category')
            ax_bar.set_xlabel('Product Category')
            ax_bar.set_ylabel('Average Purchase Amount ($)')
            ax_bar.tick_params(axis='x', rotation=45)

            # If both are shown, bar goes left; if only bar, still left
            with col_left:
                st.pyplot(fig_bar, use_container_width=True)

        # --- Pie chart (Top 3) ---
        if show_pie:
            top3 = grouped.head(3)

            def func(pct, allvals):
                absolute = int(round(pct / 100.0 * sum(allvals)))
                return f"{pct:.1f}%\n({absolute})"

            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            colors = ['#FF5733', 'lightgreen', 'skyblue'][:len(top3)]
            ax_pie.pie(
                top3.values,
                labels=[f"{cat} $" for cat in top3.index],
                autopct=lambda pct: func(pct, top3.values),
                startangle=90,
                colors=colors,
                shadow=True,
                labeldistance=1.09
            )
            ax_pie.set_title("Top Three Product Categories Based on Purchases", fontsize=14, pad=30)
            ax_pie.axis('equal')

            # If bar is shown, pie goes right; otherwise put pie on the left
            target_col = col_right if show_bar else col_left
            with target_col:
                st.pyplot(fig_pie, use_container_width=True)

def page_discount_analysis(df: pd.DataFrame):
    st.markdown("<h2 style='text-align: left;'>Average Purchase Amount Based on Discount Usage</h2>", unsafe_allow_html=True)

    # Compute average purchase
    average_purchase = df.groupby('Discount Availed')['Purchase Amount ($)'].mean().round(2)
    average_purchase.index = ['Discount Didn\'t Use', 'Discount Used']
    discount = pd.Series(average_purchase.index)
    avg = pd.Series(average_purchase.values)

    # --- Top row: metrics + checkbox ---
    col13, col14, col15 = st.columns([1, 1, 1])
    with col13:
        st.metric(label=discount[0], value=f"${avg[0]:,.2f}")
    with col14:
        st.metric(label=discount[1], value=f"${avg[1]:,.2f}")
    with col15:
        st.markdown("#### 📊 Visualizations")
        st.markdown(
            "<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
            unsafe_allow_html=True
        )
        show_bar = st.checkbox(label='Show Bar Chart', key='bar_box2')

    if show_bar:
        colors = ['salmon', 'skyblue']
        # Smaller figure so it doesn't look huge if you choose not to expand to container width
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(discount, avg, color=colors, edgecolor='black')

        ax.set_title("Discount vs Didn't Discount Users", fontsize=16, pad=20)
        ax.set_xlabel("Discount Category", fontsize=12)
        ax.set_ylabel("Average Purchase Amount ($)", fontsize=12)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"${bar.get_height():,.2f}",
                ha='center', va='bottom', fontsize=10
            )

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()

        # NEW ROW: two equal columns; render only in the left one
        left_col, right_col = st.columns([1, 1], gap="large")
        with left_col:
            # Option 1: fill just the left column (recommended)
            st.pyplot(fig, use_container_width=True)

            # Option 2 (if you still find it too wide): don't expand, rely on figsize
            # st.pyplot(fig, use_container_width=False)

def page_satisfied_return_customers(df: pd.DataFrame):
    satisfied_returns = df[(df['Review Score (1-5)'] >= 4) & (df['Return Customer'] == True)]
    percentage = (len(satisfied_returns) / len(df)) * 100 if len(df) else 0

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        st.markdown("#### 📊 Satisfied Return Customers")
        st.write(f"**Percentage of satisfied return customers:** {percentage:.2f}%")
        st.markdown("""
        - **Only 20.08% of customers are satisfied return customers** — meaning just 1 in 5 customers who return to shop are highly satisfied (review score ≥ 4).
        - **This highlights a critical retention gap**: most returning customers are not experiencing a consistently satisfying experience, which threatens long-term loyalty and revenue.
        """)

    with col_right:
        colors = ['#FF5733', 'lightgreen']
        labels = ['Satisfied Return Customers', 'Others']
        sizes = [len(satisfied_returns), len(df) - len(satisfied_returns)]
        explodes = [0.1, 0]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            startangle=10,
            shadow=True,
            explode=explodes,
            textprops={'fontsize': 10},
        )
        ax.axis('equal')
        ax.set_title('Satisfied Return Customers')
        st.pyplot(fig)

def page_time_vs_purchase(df: pd.DataFrame):
    st.markdown("### 🕒 Time Spent VS Purchase Amount")
    correlation = df['Time Spent on Website (min)'].corr(df['Purchase Amount ($)'])

    col1, col2 = st.columns([2, 2])
    with col1:
        st.markdown("#### 🔗 Correlation")
        st.markdown(f"""
        <div style='font-size:40px; font-weight:bold; color:#1f77b4;'>
            {round(correlation, 3)}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        ★ There is almost no linear relationship between time spent on the website and purchase amount.  
        ★ The correlation is **+0.01**, which is extremely weak and nearly negligible.  
        ★ This means:  
        &nbsp;&nbsp;• Customers who spend more time on the website are **not significantly** more likely to spend more money.
        """, unsafe_allow_html=True)

    with col2:
        fig = px.density_heatmap(
            df,
            x='Time Spent on Website (min)',
            y='Purchase Amount ($)',
            nbinsx=30,
            nbinsy=30,
            color_continuous_scale='Blues',
            labels={'Time Spent on Website (min)': 'Time Spent (min)', 'Purchase Amount ($)': 'Purchase ($)'},
            title='Interactive Hexbin: Time Spent vs. Purchase Amount'
        )
        fig.update_traces(hovertemplate='Time: %{x}<br>Purchase: %{y}<br>Density: %{z}<extra></extra>')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 7) Router
# -------------------------
ROUTER = {
    "📊 Age Stats": lambda: page_age_stats(df),
    "💰 Purchase Amount Stats": lambda: page_purchase_amount_stats(df),
    "📱 Device Type Stats": lambda: page_device_type_stats(df),
    "🛍️ Product Category Stats": lambda: page_product_category_stats(df),
    "🎟️ Discount Analysis": lambda: page_discount_analysis(df),
    "✅ Satisfied Return Customers": lambda: page_satisfied_return_customers(df),
    "🕒 Time Spent vs Purchase": lambda: page_time_vs_purchase(df)
}

# Render selected page
ROUTER[page]()

st.sidebar.divider()  # Adds a line + space


# Key Metrics
st.sidebar.title("Key Metrics")
st.sidebar.markdown(
    "<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
    unsafe_allow_html=True
)

total_customers = len(df)
avg_purchase = df['Purchase Amount ($)'].mean()
avg_review_score = df['Review Score (1-5)'].mean()
avg_delivery_time = df['Delivery Time (days)'].mean()
common_payment = df['Payment Method'].mode()[0] if 'Payment Method' in df.columns else 'N/A'
avg_item_purchase = df['Number of Items Purchased'].mean()
avg_age = df['Age'].mean()
avg_time_spent = df['Time Spent on Website (min)'].mean()


st.sidebar.markdown(f"""
- **Total Customers**: {total_customers}
- **Avg Purchase Amount**: ${avg_purchase:.2f}
- **Avg Review Score**: {avg_review_score:.1f}
- **Avg Delivery Time**: {avg_delivery_time:.1f} days
- **Common Payment Method**: {common_payment}
- **Avg Items Purchased**: {avg_item_purchase:.2f}
- **Avg Age**: {avg_age:.2f} years
- **Avg Time Spent on Website**: {avg_time_spent:.2f} minutes
""")

st.sidebar.divider()  # Adds a line + space

# Creator Profile
st.sidebar.markdown("### 👤 Creator Profile")
st.sidebar.markdown(
    "<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
    unsafe_allow_html=True
)
st.sidebar.markdown("""
**Name:** Mir Shahadut Hossain  
**Email:** sujon6901@gmail.com  
**Contact:** +8801671761312  
**GitHub:** [github.com/doyancha](https://github.com/doyancha)
""")
st.sidebar.divider()  # Adds a line + space



st.divider()  # Adds a line + space

def style_cells_random(df):
    import random
    def random_color(_):
        return f'background-color: #{random.randint(0, 0xFFFFFF):06x}'
    return df.style.applymap(random_color)


# Create Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Payment Method Frequency",
    "⭐ Avg. Review Scores",
    "📦 Items vs Satisfaction",
    "📍 Purchase by Location",
    "🔄 Return vs Non-Return",
    "💳 Payment Method Analysis"
])

# ==================================================================================
# TAB 1: Payment Method Frequency
# ==================================================================================
with tab1:
    method_counts = df['Payment Method'].value_counts()
    payment_method_series = pd.Series(method_counts.index, name='Payment Method')
    total_payment_series = pd.Series(method_counts.values, name='Total Users')

    payment_method_table = pd.DataFrame({
        'Payment Method': payment_method_series,
        'Total Users': total_payment_series
    })

    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("#### 📋 Most Common Payment Method Used by Customers")
        st.dataframe(
            style_cells_random(payment_method_table).format({"Value": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )
    with col_right:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000;'>", unsafe_allow_html=True)
        show_chart = st.checkbox("Show Bar Chart", key="payment_chart_checkbox")

    if show_chart:
        total_sum = total_payment_series.sum()
        col_left, col_center, col_right = st.columns([2, 1, 1])
        with col_left:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=payment_method_series,
                y=total_payment_series,
                palette='Set2',
                edgecolor='black',
                ax=ax
            )
            for bar in ax.patches:
                height = bar.get_height()
                percentage = (height / total_sum) * 100 if total_sum else 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height*0.5,
                    f"{height:,.0f}\n({percentage:.1f}%)",
                    ha='center', va='bottom', fontsize=10
                )
            ax.set_title("Top Payment Methods", fontsize=16)
            ax.set_xlabel("Payment Method", fontsize=14)
            ax.set_ylabel("Total Users", fontsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)

# ==================================================================================
# TAB 2: Avg. Review Scores by Payment Method
# ==================================================================================
with tab2:
    avg_review_scores = df.groupby('Payment Method')['Review Score (1-5)'].mean().round(2)
    avg_review_scores_df = avg_review_scores.reset_index()
    avg_review_scores_df.columns = ['Payment Method', 'Average Review Score']

    col_table, col_control = st.columns([3, 1])
    with col_table:
        st.markdown("#### ⭐ Average Review Scores by Payment Method")
        st.dataframe(
            style_cells_random(avg_review_scores_df).format({"Average Review Score": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )
    with col_control:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000;'>", unsafe_allow_html=True)
        show_plot = st.checkbox("Show Bar Plot", key='avg_review_barplot')

    if show_plot:
        col_left, col_center, col_right = st.columns([2, 1, 1])
        with col_left:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=avg_review_scores_df,
                x='Payment Method',
                y='Average Review Score',
                palette='Set2',
                edgecolor='black',
                ax=ax
            )
            for bar in ax.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}",
                        ha='center', va='bottom', fontsize=10)
            ax.set_title("Payment Method vs Avg. Review Score", fontsize=16)
            ax.set_xlabel("Payment Method", fontsize=14)
            ax.set_ylabel("Avg. Review Score", fontsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)

# ==================================================================================
# TAB 3: Items Purchased vs. Satisfaction (+ Animated Plotly)
# ==================================================================================
with tab3:
    st.subheader("📦 Items Purchased vs. Customer Satisfaction")
    # Static seaborn bar (centered half width)
    col_left, col_right = st.columns([1, 2])
    with col_left:
        correlation1 = df['Number of Items Purchased'].corr(df['Customer Satisfaction Numeric'])
        st.markdown("#### 🔗 Correlation")
        st.markdown(f"<div style='font-size:40px; font-weight:bold; color:#1f77b4;'>{correlation1:.2f}</div>",
                    unsafe_allow_html=True)
        st.caption("Correlation between items purchased and satisfaction (Low=1, Medium=2, High=3).")

        st.markdown("""
        ★ Virtually no linear relationship: customers buying more items are not necessarily more satisfied.  
        ★ These variables behave independently in this dataset.
        """)
    with col_right:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=df,
            x='Customer Satisfaction',
            y='Number of Items Purchased',
            order=['Low', 'Medium', 'High'],
            palette='Blues',
            ax=ax
        )
        ax.set_title('Average Items Purchased by Satisfaction Level', fontsize=14)
        ax.set_xlabel('Customer Satisfaction', fontsize=12)
        ax.set_ylabel('Average Items Purchased', fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)

    # Animated Plotly scatter (optional)
    st.markdown("#### 🎞️ Animated: Time vs Purchase by Satisfaction")
    show_anim_scatter = st.checkbox("Show animated scatter", value=False, key="anim_tab3_scatter")
    if show_anim_scatter and {'Time Spent on Website (min)', 'Purchase Amount ($)', 'Customer Satisfaction'}.issubset(df.columns):
        figp = px.scatter(
            df,
            x='Time Spent on Website (min)',
            y='Purchase Amount ($)',
            animation_frame='Customer Satisfaction',
            color='Return Customer' if 'Return Customer' in df.columns else None,
            size='Number of Items Purchased' if 'Number of Items Purchased' in df.columns else None,
            hover_data=df.columns,
            title="Time vs Purchase — animated by Satisfaction"
        )
        figp.update_layout(height=520)
        st.plotly_chart(figp, use_container_width=True)

# ==================================================================================
# TAB 4: Average Purchase by Location
# ==================================================================================
with tab4:
    avg_purchase_by_location = df.groupby('Location')['Purchase Amount ($)'].mean()
    sorted_avg = avg_purchase_by_location.sort_values(ascending=False)

    table_df = pd.DataFrame({
        'Location': sorted_avg.index,
        'Average Purchase Amount ($)': sorted_avg.values.round(2)
    })

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 📍 Average Purchase by Location")
        st.dataframe(
            style_cells_random(table_df).format({"Average Purchase Amount ($)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )
    with col2:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000;'>", unsafe_allow_html=True)
        show_chart = st.checkbox("Show bar chart", value=False)

    if show_chart:
        df_plot = table_df.copy()
        second_highest = df_plot.iloc[1]['Location'] if len(df_plot) >= 2 else None
        colors = ['orange' if loc == second_highest else 'lightblue' for loc in df_plot['Location']]

        col_left, col_center, col_right = st.columns([2, 1, 1])
        with col_left:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=df_plot,
                x='Location',
                y='Average Purchase Amount ($)',
                palette=colors,
                edgecolor='black',
                ax=ax
            )
            max_val = df_plot['Average Purchase Amount ($)'].max()
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2, height + (0.01 * max_val), f"{height:.2f}",
                        ha='center', va='bottom', fontsize=10)
            ax.set_title("Average Purchase Amount by Location", fontsize=16, pad=20)
            ax.set_xlabel("Location", fontsize=12)
            ax.set_ylabel("Purchase Amount ($)", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)

# ==================================================================================
# TAB 5: Return vs Non-Return Customers Summary (+ Animated Plotly)
# ==================================================================================
with tab5:
    returners = df[df['Return Customer'] == True] if 'Return Customer' in df.columns else df.iloc[0:0]
    non_returners = df[df['Return Customer'] == False] if 'Return Customer' in df.columns else df.iloc[0:0]

    columns_to_compare = [
        c for c in [
            'Purchase Amount ($)', 'Time Spent on Website (min)',
            'Number of Items Purchased', 'Review Score (1-5)',
            'Delivery Time (days)', 'Customer Satisfaction Numeric'
        ] if c in df.columns
    ]

    summary = pd.DataFrame({
        'Return Customers': returners[columns_to_compare].mean(),
        'Non-Return Customers': non_returners[columns_to_compare].mean()
    })
    summary['Difference'] = summary['Return Customers'] - summary['Non-Return Customers']
    summary_display = summary.sort_values(by='Return Customers', ascending=False).round(2).reset_index()
    summary_display.rename(columns={'index': 'Metric'}, inplace=True)

    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.subheader("Return vs. Non-Return Customers — Summary")
        num_cols = summary_display.select_dtypes(include=['number']).columns

        styled = (
            style_cells_random(summary_display)
            .format("{:.2f}", subset=num_cols)
            )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True
            )
        st.caption("“Difference” = Return Customers − Non-Return Customers (positive means higher for returners).")
    with col2:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000;'>", unsafe_allow_html=True)
        show_plot = st.checkbox("Show comparison plot", value=False, key="return_vs_plot")
        show_anim = st.checkbox("🎞️ Show animated metrics (Plotly)", value=False, key="anim_tab5")

    if show_plot and not summary.empty:
        sorted_summary = summary.sort_values(by='Return Customers', ascending=False)
        idx = np.arange(len(sorted_summary))
        bar_width = 0.4

        col_left, col_center, col_right = st.columns([2, 1, 1])
        with col_left:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(idx + bar_width / 2, sorted_summary['Return Customers'], height=bar_width,
                    label='Return Customers', color='#4CAF50')
            ax.barh(idx - bar_width / 2, sorted_summary['Non-Return Customers'], height=bar_width,
                    label='Non-Return Customers', color='#FF9800')
            ax.set_yticks(idx)
            ax.set_yticklabels(sorted_summary.index, fontsize=10)
            ax.set_xlabel('Average Value', fontsize=12)
            ax.set_title('Return vs Non-Return Customers: Key Metrics', fontsize=14, weight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

    # Animated Plotly bar: frame by Metric (swaps through metrics)
    if show_anim and not summary.empty:
        long_df = (
            summary.reset_index()
                   .rename(columns={'index': 'Metric'})
                   .melt(id_vars='Metric',
                         value_vars=['Return Customers', 'Non-Return Customers'],
                         var_name='Group',
                         value_name='Value')
        )
        xr = [0, float(long_df['Value'].max()) * 1.15] if long_df['Value'].notna().any() else [0, 1]
        figp = px.bar(
            long_df,
            y='Group', x='Value',
            animation_frame='Metric',
            orientation='h',
            title='Animated: Return vs Non-Return by Metric'
        )
        figp.update_layout(height=520, xaxis_range=xr)
        st.plotly_chart(figp, use_container_width=True)

# ==================================================================================
# TAB 6: Payment Method Analysis (Return Rate & Satisfaction + Animated)
# ==================================================================================
with tab6:
    # Return Rate
    if 'Payment Method' in df.columns and 'Return Customer' in df.columns:
        return_rate_by_payment = df.groupby('Payment Method')['Return Customer'].mean().sort_values(ascending=False)
        df_return = (return_rate_by_payment * 100).reset_index()
        df_return.columns = ['Payment Method', 'Return Rate (%)']
    else:
        df_return = pd.DataFrame(columns=['Payment Method', 'Return Rate (%)'])
        return_rate_by_payment = pd.Series(dtype=float)

    # Satisfaction
    if 'Customer Satisfaction Numeric' not in df.columns and 'Customer Satisfaction' in df.columns:
        satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['Customer Satisfaction Numeric'] = df['Customer Satisfaction'].map(satisfaction_map)
    if 'Payment Method' in df.columns and 'Customer Satisfaction Numeric' in df.columns:
        satisfaction_by_payment = df.groupby('Payment Method')['Customer Satisfaction Numeric'].mean().sort_values(ascending=False)
        satisfaction_df = satisfaction_by_payment.round(2).reset_index()
        satisfaction_df.columns = ['Payment Method', 'Avg. Satisfaction Score']
    else:
        satisfaction_by_payment = pd.Series(dtype=float)
        satisfaction_df = pd.DataFrame(columns=['Payment Method', 'Avg. Satisfaction Score'])

    # Return Rate table + checkbox
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.subheader("Return Rate by Payment Method (Highest to Lowest)")
        num_cols_return = df_return.select_dtypes(include=["number"]).columns
        styled_return = (
            style_cells_random(df_return)
            .format("{:.2f}", subset=num_cols_return)
    )

        st.dataframe(
            styled_return,
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                    unsafe_allow_html=True)
        show_return_plot = st.checkbox("Show Return Rate Plot", value=False, key="pm_return_rate_plot")

    # Satisfaction table + checkbox
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.subheader("Average Satisfaction Score by Payment Method")
        

        num_cols_return = satisfaction_df.select_dtypes(include=["number"]).columns
        styled_return = (
            style_cells_random(satisfaction_df)
            .format("{:.2f}", subset=num_cols_return)
        )

        st.dataframe(
            styled_return,
            use_container_width=True,
            hide_index=True
        )
    with col2:
        st.markdown("#### 📊 Visualizations")
        st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                    unsafe_allow_html=True)
        show_satisfaction_plot = st.checkbox("Show Satisfaction Plot", value=False, key="pm_satisfaction_plot")

    # Unified plotting row: left (return rate) | right (satisfaction)
    if show_return_plot or show_satisfaction_plot:
        left_col, right_col = st.columns(2)

        if show_return_plot and not df_return.empty:
            with left_col:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    data=df_return.sort_values('Return Rate (%)'),
                    y='Payment Method',
                    x='Return Rate (%)',
                    palette='Blues_d',
                    ax=ax
                )
                ax.set_title('Return Rate by Payment Method', fontsize=14, weight='bold')
                ax.set_xlabel('Return Customer Rate (%)', fontsize=12)
                ax.set_ylabel('')
                ax.xaxis.grid(False); ax.yaxis.grid(False)
                sns.despine(left=True, top=True, right=True)
                for p in ax.patches:
                    width = p.get_width()
                    ax.text(width + 0.5,
                            p.get_y() + p.get_height() / 2,
                            f'{width:.1f}%',
                            va='center', ha='left', fontsize=10)
                fig.tight_layout()
                st.pyplot(fig)

        if show_satisfaction_plot and not satisfaction_df.empty:
            with right_col:
                sorted_satisfaction = satisfaction_by_payment.sort_values()
                df_plot = sorted_satisfaction.reset_index()
                df_plot.columns = ['Payment Method', 'Satisfaction']

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    data=df_plot,
                    y='Payment Method',
                    x='Satisfaction',
                    palette='Greens_r',
                    ax=ax
                )
                ax.set_title('Avg. Customer Satisfaction by Payment Method', fontsize=14, weight='bold')
                ax.set_xlabel('Average Satisfaction Score', fontsize=12)
                ax.set_ylabel('')
                ax.xaxis.grid(False); ax.yaxis.grid(False)
                sns.despine(left=True, top=True, right=True)
                for p in ax.patches:
                    width = p.get_width()
                    ax.text(width + 0.02,
                            p.get_y() + p.get_height() / 2,
                            f'{width:.2f}',
                            va='center', ha='left', fontsize=10)
                fig.tight_layout()
                st.pyplot(fig)

    # Animated Plotly combined (two-frame swap: Return Rate vs Satisfaction)
    st.markdown("#### 🎞️ Animated: Payment Metrics by Method (Plotly)")
    show_anim_tab6 = st.checkbox("Show animated combined chart", value=False, key="anim_tab6")
    if show_anim_tab6 and (not df_return.empty or not satisfaction_df.empty):
        combo_frames = []
        if not df_return.empty:
            rr = df_return.rename(columns={'Return Rate (%)': 'Value'}).copy()
            rr['Metric'] = 'Return Rate (%)'
            combo_frames.append(rr[['Payment Method', 'Value', 'Metric']])
        if not satisfaction_df.empty:
            ss = satisfaction_df.rename(columns={'Avg. Satisfaction Score': 'Value'}).copy()
            ss['Metric'] = 'Avg. Satisfaction'
            combo_frames.append(ss[['Payment Method', 'Value', 'Metric']])

        if combo_frames:
            combined = pd.concat(combo_frames, ignore_index=True)
            xr = [0, float(combined['Value'].max()) * 1.15] if combined['Value'].notna().any() else [0, 1]
            figp = px.bar(
                combined,
                y='Payment Method', x='Value',
                color='Payment Method',
                animation_frame='Metric',
                orientation='h',
                title='Animated: Return Rate vs Satisfaction by Payment Method'
            )
            figp.update_layout(height=640, xaxis_range=xr, showlegend=False)
            st.plotly_chart(figp, use_container_width=True)



st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line

st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
            unsafe_allow_html=True)


# ================== TABLE (LEFT) ==================
# Calculate the location stats
location_stats = (
    df.groupby('Location')[['Purchase Amount ($)', 'Delivery Time (days)']]
    .mean()
    .sort_values(by='Purchase Amount ($)', ascending=False)
    .round(2)
)

# Styled table
styled = (
    location_stats.style
        .format({
            'Purchase Amount ($)': '{:.2f}',
            'Delivery Time (days)': '{:.2f}'
        })
        .background_gradient(subset=['Purchase Amount ($)'], cmap='YlGn')   # green gradient
        .background_gradient(subset=['Delivery Time (days)'], cmap='Blues') # blue gradient
        .set_table_styles([
            {"selector": "thead th",
             "props": [("background-color", "#333"),
                       ("color", "white"),
                       ("font-size", "14px")]}
        ])
)

# ================== LAYOUT ==================
col_table, col_plots = st.columns([6.5, 2], gap="large")

with col_table:
    st.subheader("Average Purchase Amount and Delivery Time by Location")
    st.dataframe(styled, use_container_width=True)

with col_plots:
    st.markdown("#### 📊 Visualizations")
    st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                        unsafe_allow_html=True)
    # -------- Plot 1: Purchase Amount by Location with Delivery Speed (checkbox) --------
    show_plot1 = st.checkbox("Show: Avg Purchase Amount by Location", value=False, key="plot_loc_pa")

    if show_plot1:
        # Prepare delivery speed bins/labels
        bins = [0, 6.9, 7.05, 8]
        labels = ['Fast', 'Moderate', 'Slow']
        df['Delivery Speed'] = pd.cut(df['Delivery Time (days)'], bins=bins, labels=labels, include_lowest=True)

        # Plot 1
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        sns.set_style("white")

        sns.barplot(
            data=df,
            x='Location',
            y='Purchase Amount ($)',
            hue='Delivery Speed',
            hue_order=labels,
            palette='Blues',
            ax=ax1
        )

        ax1.set_title('Average Purchase Amount by Location with Delivery Time', fontsize=14, weight='bold')
        ax1.set_xlabel('Location', fontsize=12)
        ax1.set_ylabel('Purchase Amount ($)', fontsize=12)

        # Ticks and layout
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.tick_params(axis='y', labelsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        fig1.tight_layout()

        st.pyplot(fig1)
    st.markdown("#### 📊 Visualizations")
    st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                        unsafe_allow_html=True)
    # -------- Plot 2: Average Delivery Time by Location (checkbox) --------
    show_plot2 = st.checkbox("Show: Avg Delivery Time by Location", value=False, key="plot_loc_dt")

    if show_plot2:
        # Prepare DataFrame from location_stats for seaborn
        df_dt = location_stats.reset_index()

        # Plot 2
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        sns.set_style("white")

        sns.barplot(
            data=df_dt,
            x='Location',
            y='Delivery Time (days)',
            palette='Reds',
            ax=ax2
        )

        ax2.set_title('Average Delivery Time by Location', fontsize=16, weight='bold')
        ax2.set_xlabel('Location', fontsize=12)
        ax2.set_ylabel('Delivery Time (days)', fontsize=12)

        # Ticks and layout
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.tick_params(axis='y', labelsize=10)
        # Optional grid:
        # ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Custom legend (single patch)
        red_patch = mpatches.Patch(color='tomato', label='Delivery Time (days)')
        ax2.legend(handles=[red_patch], loc='upper right', bbox_to_anchor=(1.0, 1.01))

        fig2.tight_layout()
        st.pyplot(fig2)


# --- Checkbox to show the plots ---
show_plot = st.checkbox("Show Purchase Amount & Delivery Time Comparison Plots")

if show_plot:
    # Prepare delivery speed categories
    bins = [0, 6.9, 7.05, 8]
    labels = ['Fast', 'Moderate', 'Slow']
    df['Delivery Speed'] = pd.cut(df['Delivery Time (days)'], bins=bins, labels=labels)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.set_style("whitegrid")

    # --- Subplot 1: Purchase Amount by Location with Delivery Speed ---
    sns.barplot(
        data=df,
        x='Location',
        y='Purchase Amount ($)',
        hue='Delivery Speed',
        palette='Blues',
        ax=ax1
    )
    ax1.set_title('Avg. Purchase Amount by Location / Delivery Speed', fontsize=14, weight='bold')
    ax1.set_xlabel('Location', fontsize=12)
    ax1.set_ylabel('Purchase Amount ($)', fontsize=12)
    ax1.tick_params(axis='x', labelsize=10, rotation=30)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title='Delivery Speed', loc='upper right')

    # --- Subplot 2: Delivery Time by Location ---
    sns.barplot(
        x=location_stats.index,
        y=location_stats['Delivery Time (days)'],
        palette='Reds',
        hue=location_stats.index,
        legend=False,
        ax=ax2
    )
    ax2.set_title('Average Delivery Time by Location', fontsize=14, weight='bold')
    ax2.set_xlabel('Location', fontsize=12)
    ax2.set_ylabel('Delivery Time (days)', fontsize=12)
    ax2.tick_params(axis='x', labelsize=10, rotation=30)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Custom legend for Subplot 2
    red_patch = mpatches.Patch(color='tomato', label='Delivery Time (days)')
    ax2.legend(handles=[red_patch], loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)



st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line


st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
                        unsafe_allow_html=True)




# (Optional) Header image if you want to show it above the summary
#st.image("/mnt/data/090a9021-e6be-4496-910c-df7d9e4e758a.png", use_container_width=True)

# The summary content as a string (so we can also download it)
insights_md = """
## **📊 Key Insights from Customer Analysis**

---

### **🚚 Customer Satisfaction is Highly Influenced by Delivery Time**
- Customers who received products quickly gave **higher review scores**.  
- As delivery time increases, **customer satisfaction drops noticeably**.  
- Efficient logistics are **key to improving customer experience**, especially for first-time buyers.

---

### **🔁 Returning Customers Show Stronger Loyalty**
- Returning customers provide **higher review scores** and **spend more**.  
- They are more trusting and loyal → a valuable segment for **loyalty programs & personalized offers**.

---

### **📦 Top Revenue-Generating Product Categories**
- **Toys, Books, Home Essentials** drive a significant share of sales.  
- Increase **visibility & stock** of these categories to boost revenue, especially in peak seasons.

---

### **💳 Payment Method Usage vs. Satisfaction**
- **Bank Transfer** is most used but doesn’t have the highest satisfaction.  
- **Cash on Delivery** users report slightly better satisfaction.  
- Promoting **secure, flexible payment options (e.g., mobile wallets)** could enhance trust.

---

### **📍 Location-Wise Purchase Behavior**
- **Dhaka, Chittagong, Khulna** customers spend the most.  
- Urban areas enjoy **shorter delivery times & higher review scores**.  
- Region‑specific **marketing & logistics optimization** is needed.

---

### **🔔 Subscription Status Boosts Spending & Satisfaction**
- Subscribers spend **more frequently & in larger amounts**.  
- Tiered subscription programs can **increase revenue & loyalty**.

---

### **📱 Device Type & Gender Influence Behavior**
- **Mobile users** spend less time but complete purchases efficiently.  
- Gender shows some variation in product preferences → good for **targeted campaigns**.

---

### **🏷️ Discounts Increase Purchase Volume**
- Discount users buy **more items** and show slightly **higher satisfaction**.  
- Strategic, **time-limited or personalized discounts** can drive both sales & loyalty.

---
"""

# Expander with the content
with st.expander("📑 Show Key Insights Summary", expanded=False):
    st.markdown(insights_md)

    # (Optional) let users download the summary as a Markdown file
    st.download_button(
        "⬇️ Download summary (Markdown)",
        data=insights_md,
        file_name="key_insights_summary.md",
        mime="text/markdown",
        use_container_width=True
    )


st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line

st.markdown("<hr style='border: 3px solid #000000; margin-top: -10px; margin-bottom: 20px;'>",
            unsafe_allow_html=True)

st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line
st.write("")   # Adds one blank line