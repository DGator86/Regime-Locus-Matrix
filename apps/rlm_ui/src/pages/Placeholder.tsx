interface Props { title: string; description?: string }

export default function Placeholder({ title, description }: Props) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center p-8">
      <div className="text-4xl font-black text-slate-800">{title}</div>
      <p className="text-sm text-slate-600 max-w-sm">
        {description ?? 'This page is coming soon.'}
      </p>
    </div>
  )
}
